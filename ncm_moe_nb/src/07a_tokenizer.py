# %% [markdown]
# ## 4b. Learned tokenizers: multi-scale residual VQ-VAE for images and audio
#
# The frozen k-means codecs cap image fidelity near 20 dB (a blocky mosaic) and give robotic audio. These replace them
# with small **learned** tokenizers that fit the same ladder, so the model, the layouts and the vocabulary (257) do not change:
#
# * a convolutional **encoder** maps the image (or the log-mel) to a latent grid at the finest ladder resolution (16x16, or 512 frames);
# * **multi-scale residual quantisation** (VAR style): level 0 quantises the average of the latent, every next level quantises
#   what the levels before it missed, one 256-entry codebook per level (EMA-updated, dead codes re-seeded);
# * a convolutional **decoder** rebuilds the image / mel from the sum of the quantised levels, so decoding levels 0..k gives a
#   progressively sharper result (trained on random prefixes as well as the full stack).
#
# Because the codes are residual, decoding level *k* needs the codes of **all** levels <= *k*: `decode(codes_list, level=k)`.
# They are trained once at startup on a streamed sample held in RAM (same idea as the old codebook fit) and the weights go into the
# checkpoint. **Gate:** before any model compute is spent, each learned tokenizer is compared with a k-means codec fitted on the same
# data; if the image tokenizer is not `TOK_GATE_DB` dB better (audio: 5% lower log-mel error) the notebook stops.
# The bit budget is unchanged (8 bits per code), so expect a real gain from the decoder and the residual structure,
# not a different league; much higher fidelity needs more codes per image.

# %%
def _gn(c):
    return nn.GroupNorm(8, c)


class _Res2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1, self.n2, self.c2 = _gn(c), nn.Conv2d(c, c, 3, padding=1), _gn(c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class _Res1d(nn.Module):
    def __init__(self, c, dil=1):
        super().__init__()
        self.n1, self.c1 = _gn(c), nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.n2, self.c2 = _gn(c), nn.Conv1d(c, c, 5, padding=2)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class EMAQuant(nn.Module):
    """One 256-entry codebook updated by exponential moving averages (no codebook gradient); dead codes are re-seeded from the batch."""

    def __init__(self, n=256, d=64, decay=0.99, eps=1e-5):
        super().__init__()
        self.n, self.d, self.decay, self.eps = n, d, decay, eps
        self.register_buffer("embed", torch.randn(n, d) * 0.5)
        self.register_buffer("size", torch.ones(n))
        self.register_buffer("avg", self.embed.clone())
        self.register_buffer("steps", torch.zeros(()))
        self.register_buffer("seeded", torch.zeros(()))

    def forward(self, x):
        """x [N, d] -> (quantised, straight-through [N, d]; index [N]; commitment mse)."""
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            if self.training and float(self.seeded) == 0:                              # first batch: start the codebook on real data
                pick = torch.randint(x.shape[0], (self.n,), device=x.device)
                self.embed.copy_(x[pick].detach() + 0.01 * torch.randn(self.n, self.d, device=x.device))
                self.avg.copy_(self.embed); self.seeded.fill_(1)
            dist = (x * x).sum(1, keepdim=True) - 2 * x @ self.embed.T + (self.embed * self.embed).sum(1)
            idx = dist.argmin(1)
            q = self.embed[idx]
            if self.training:
                with torch.no_grad():
                    oh = F.one_hot(idx, self.n).float()
                    self.size.mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
                    self.avg.mul_(self.decay).add_(oh.T @ x.detach(), alpha=1 - self.decay)
                    tot = self.size.sum()
                    sz = (self.size + self.eps) / (tot + self.n * self.eps) * tot
                    self.embed.copy_(self.avg / sz[:, None])
                    self.steps += 1
                    if int(self.steps) % 25 == 0:                                      # re-seed codes used < 5% as often as an average code
                        dead = self.size < 0.05 * tot / self.n
                        nd = int(dead.sum())
                        if nd:
                            r = x.detach()[torch.randint(x.shape[0], (nd,), device=x.device)]
                            self.embed[dead] = r; self.avg[dead] = r; self.size[dead] = tot / self.n
            commit = F.mse_loss(x, q.detach())
            q = x + (q - x).detach()
        return q, idx, commit


class MSRVQ(nn.Module):
    """Multi-scale residual quantiser. `sizes` = grid side per level (2D) or length per level (1D); input latent [B, d, *finest]."""

    def __init__(self, sizes, d, nd):
        super().__init__()
        self.sizes, self.d, self.nd = list(sizes), d, nd
        self.q = nn.ModuleList(EMAQuant(256, d) for _ in self.sizes)

    def _pool(self, r, n):
        return F.adaptive_avg_pool2d(r, n) if self.nd == 2 else F.adaptive_avg_pool1d(r, n)

    def _up(self, q, size):
        return (F.interpolate(q, size=(size, size), mode="bilinear", align_corners=False) if self.nd == 2
                else F.interpolate(q, size=size, mode="linear", align_corners=False))

    def forward(self, z):
        """z [B, d, ...] float -> (summed quantised latent after each level [K], 1-based codes per level [K x [B, n^nd]], commitment loss)."""
        fin = z.shape[-1]
        f_hat, outs, codes, commit = torch.zeros_like(z), [], [], 0.0
        for n, qz in zip(self.sizes, self.q):
            rp = self._pool(z - f_hat, n)                                               # what is still missing, at this level's resolution
            flat = rp.permute(0, 2, 3, 1).reshape(-1, self.d) if self.nd == 2 else rp.permute(0, 2, 1).reshape(-1, self.d)
            qf, idx, c = qz(flat)
            qg = qf.view(rp.shape[0], *rp.shape[2:], self.d)
            qg = qg.permute(0, 3, 1, 2) if self.nd == 2 else qg.permute(0, 2, 1)
            f_hat = f_hat + self._up(qg, fin)
            outs.append(f_hat); codes.append(idx.view(rp.shape[0], -1) + 1); commit = commit + c
        return outs, codes, commit

    @torch.no_grad()
    def from_codes(self, codes, upto):
        """1-based codes (list with >= upto+1 entries) -> summed quantised latent after levels 0..upto, at the finest size."""
        fin, f_hat = self.sizes[-1], 0.0
        for k in range(upto + 1):
            n = self.sizes[k]
            e = self.q[k].embed[(codes[k] - 1).clamp(min=0)]                            # [B, n^nd, d]
            e = e.view(e.shape[0], n, n, self.d).permute(0, 3, 1, 2) if self.nd == 2 else e.permute(0, 2, 1)
            f_hat = f_hat + self._up(e, fin)
        return f_hat


class ImageVQ(nn.Module):
    def __init__(self, spec, d=64):
        super().__init__()
        nd = int(round(math.log2(spec.patch)))
        assert 2 ** nd == spec.patch, f"patch {spec.patch} must be a power of two"
        ch = [min(32 * 2 ** i, 256) for i in range(nd + 1)]
        enc = [nn.Conv2d(3, ch[0], 3, padding=1)]
        for i in range(nd):
            enc += [_Res2d(ch[i]), nn.Conv2d(ch[i], ch[i + 1], 4, stride=2, padding=1)]
        enc += [_Res2d(ch[nd]), _Res2d(ch[nd]), _gn(ch[nd]), nn.SiLU(), nn.Conv2d(ch[nd], d, 1)]
        dec = [nn.Conv2d(d, ch[nd], 3, padding=1), _Res2d(ch[nd]), _Res2d(ch[nd])]
        for i in reversed(range(nd)):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch[i + 1], ch[i], 3, padding=1), _Res2d(ch[i])]
        dec += [_gn(ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 3, 3, padding=1)]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
        self.rvq = MSRVQ([math.isqrt(n) for n in spec.lens("image")], d, 2)


class AudioVQ(nn.Module):
    def __init__(self, spec, d=64, ch=256):
        super().__init__()
        m = spec.n_mels
        self.enc = nn.Sequential(nn.Conv1d(m, ch, 5, padding=2), _Res1d(ch, 1), _Res1d(ch, 2), _Res1d(ch, 4), _Res1d(ch, 8),
                                 _gn(ch), nn.SiLU(), nn.Conv1d(ch, d, 1))
        self.dec = nn.Sequential(nn.Conv1d(d, ch, 5, padding=2), _Res1d(ch, 1), _Res1d(ch, 2), _Res1d(ch, 4), _Res1d(ch, 8),
                                 _gn(ch), nn.SiLU(), nn.Conv1d(ch, m, 5, padding=2))
        self.rvq = MSRVQ(spec.lens("audio"), d, 1)


def _amp_bf16():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(IS_CUDA))


def _need_list(codes):
    assert isinstance(codes, (list, tuple)), "learned tokenizers decode from the code list of levels 0..k (residual codes): pass `codes`, not `codes[k]`"


class VQImageCodec:
    """Same interface as ImageCodec: encode, decode(codes_list, level), state, load. `cb` is the list of level codebooks once trained."""

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.net = ImageVQ(spec).to(device).eval()
        self.cb, self.trained = None, False

    def mark_trained(self):
        self.trained, self.cb = True, [q.embed for q in self.net.rvq.q]

    @torch.no_grad()
    def encode(self, img):
        with _amp_bf16():
            z = self.net.enc(img.float() / 127.5 - 1.0)
        return self.net.rvq(z.float())[1]

    @torch.no_grad()
    def decode(self, codes, level=None):
        _need_list(codes)
        level = self.spec.K - 1 if level is None else level
        f_hat = self.net.rvq.from_codes(codes, level)
        with _amp_bf16():
            y = self.net.dec(f_hat)
        return ((y.float() + 1.0) * 127.5).clamp(0, 255).round().to(torch.uint8)

    def state(self):
        return dict(kind="vq", trained=self.trained, sd={k: v.detach().cpu() for k, v in self.net.state_dict().items()})

    def load(self, st):
        assert st.get("kind") == "vq", "this checkpoint holds k-means codebooks but TOKENIZER='vq': use a new CKPT_DIR or TOKENIZER='kmeans'"
        self.net.load_state_dict(st["sd"])
        if st["trained"]:
            self.mark_trained()
        self.net.eval()


class VQAudioCodec(AudioCodec):
    """AudioCodec with a learned encoder/decoder over the normalised log-mel instead of k-means codebooks. The vocoder is unchanged."""

    def __init__(self, spec, device):
        super().__init__(spec, device)
        self.net = AudioVQ(spec).to(device).eval()
        self.trained = False

    def mark_trained(self):
        self.trained, self.cb = True, [q.embed for q in self.net.rvq.q]

    def norm_mel(self, wave):
        return ((self.logmel(wave) - self.mean) / self.std).transpose(1, 2)            # [B, 80, F]

    @torch.no_grad()
    def encode(self, wave):
        x = self.norm_mel(wave)                                                         # mel in fp32, outside autocast
        with _amp_bf16():
            z = self.net.enc(x)
        return self.net.rvq(z.float())[1]

    @torch.no_grad()
    def decode_mel(self, codes, level=None):
        _need_list(codes)
        level = self.spec.K - 1 if level is None else level
        f_hat = self.net.rvq.from_codes(codes, level)
        with _amp_bf16():
            y = self.net.dec(f_hat)
        return y.float().transpose(1, 2) * self.std + self.mean                         # natural-log mel [B, F, 80]

    def state(self):
        return dict(kind="vq", trained=self.trained, mean=self.mean, std=self.std,
                    sd={k: v.detach().cpu() for k, v in self.net.state_dict().items()})

    def load(self, st):
        assert st.get("kind") == "vq", "this checkpoint holds k-means codebooks but TOKENIZER='vq': use a new CKPT_DIR or TOKENIZER='kmeans'"
        self.net.load_state_dict(st["sd"]); self.mean, self.std = st["mean"], st["std"]
        if st["trained"]:
            self.mark_trained()
        self.net.eval()


# ---- training ----
def _cosine_lr(t, steps, lr, warm=100):
    return lr * min(1.0, (t + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / max(1, steps))))


def train_vq(net, batch_fn, loss_fn, steps, lr, tag):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    net.train(); t0, run = time.time(), None
    for t in range(steps):
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(t, steps, lr)
        opt.zero_grad(set_to_none=True)
        b = batch_fn()
        with _amp_bf16():
            loss, st = loss_fn(b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run = st if run is None else {k: 0.9 * run[k] + 0.1 * st[k] for k in st}
        if t == 0 or (t + 1) % max(1, steps // 8) == 0:
            print(f"    [{tag}] step {t + 1}/{steps} " + " ".join(f"{k} {v:.4f}" for k, v in run.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    net.eval()


def _recon_loss(y, x):
    return F.mse_loss(y.float(), x) + 0.2 * F.l1_loss(y.float(), x)


def _vq_loss(net, K, target):
    """target(batch) -> (input to the encoder, reconstruction target). Full stack + one random prefix, plus commitment."""
    def f(batch):
        xin, xt = target(batch)
        z = net.enc(xin).float()
        outs, _, commit = net.rvq(z)
        y = net.dec(outs[-1])
        k = random.randrange(K - 1)
        loss = 2.0 * (_recon_loss(y, xt) + 0.5 * _recon_loss(net.dec(outs[k]), xt)) + 0.25 * commit
        return loss, dict(mse=float(F.mse_loss(y.float(), xt)), commit=float(commit))
    return f


def psnr_u8(a, b):
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1).clamp(min=1e-8)
    return 10 * torch.log10(255.0 ** 2 / mse)


@torch.no_grad()
def eval_image_tok(codec, imgs):
    """mean PSNR (dB) vs the real image when decoding levels 0..K-1; works for the k-means and the learned codec."""
    per = None
    for i in range(0, len(imgs), 32):
        x = imgs[i:i + 32]
        codes = codec.encode(x)
        if per is None:
            per = [[] for _ in codes]                               # the k-means baseline has SPEC.K levels, the learned codec may have one more
        for k in range(len(per)):
            per[k].append(psnr_u8(codec.decode(codes, level=k), x))
    return [float(torch.cat(p).mean()) for p in per]


@torch.no_grad()
def eval_audio_tok(codec, wave_chunks):
    """log-mel RMSE (natural-log units) vs the real clip when decoding levels 0..K-1."""
    se, n = [0.0] * SPEC.K, 0
    for w in wave_chunks:
        target = codec.logmel(w)
        codes = codec.encode(w)
        for k in range(SPEC.K):
            se[k] += float(((codec.decode_mel(codes, level=k) - target) ** 2).mean()) * w.shape[0]
        n += w.shape[0]
    return [math.sqrt(s / n) for s in se]


@torch.no_grad()
def code_perplexity(codec, codes_fn):
    """Exp-entropy of the code histogram per level (256 = every code used equally)."""
    cnt = None
    for codes in codes_fn():
        if cnt is None:
            cnt = [torch.zeros(256, device=dev) for _ in codes]               # one histogram per level the codec returns (images may have one more than SPEC.K)
        for k, c in enumerate(codes):
            cnt[k] += torch.bincount(c.reshape(-1) - 1, minlength=256).float()
    out = []
    for c in cnt:
        p = c / c.sum().clamp(min=1)
        out.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())))
    return out


def _img_tensor(rows):
    """raw image rows -> uint8 [N, 3, S, S] on the device (nvJPEG when the row is a JPEG)."""
    out, B = [], 64
    for i in range(0, len(rows), B):
        sub, tmp = rows[i:i + B], [None] * len(rows[i:i + B])
        jp = [j for j, r in enumerate(sub) if "jpeg" in r]
        if jp:
            d = decode_jpeg_gpu([sub[j]["jpeg"] for j in jp], SPEC.side, dev)
            for n, j in enumerate(jp):
                tmp[j] = d[n]
        for j, r in enumerate(sub):
            if tmp[j] is None:
                tmp[j] = torch.from_numpy(r["img"]).to(dev)
        out.append(torch.stack(tmp))
    return torch.cat(out)


def fit_tokenizers():
    """Stream images and clips into RAM, fit a k-means baseline and train the learned tokenizers on the same data, compare, gate."""
    t0, big = time.time(), TIER != "T0"
    n_img, n_aud = TOK_IMAGES or (8192 if big else 1024), TOK_AUDIO or (2048 if big else 512)
    steps = TOK_STEPS or {"T0": 600, "T1": 4000}.get(TIER, 8000)
    K = SPEC.K
    # cache: train once, reuse on every later run with the same spec and tokenizer settings (never reused if any of them changed)
    cache = _knob("TOK_CACHE", os.path.join(CKPT_DIR, "tokenizers.pt") if CKPT_DIR else "")
    key = repr((repr(SPEC), n_img, n_aud, steps))
    if cache and os.path.exists(cache):
        st = torch.load(cache, map_location="cpu", weights_only=False)
        if st.get("key") == key:
            codecs.load(st["codecs"])
            print(f"learned tokenizers restored from {cache} (trained earlier with the same spec and settings: no refit)")
            return
        print(f"!! {cache} was trained with other settings: retraining (and overwriting it)")
    # ---------------- images ----------------
    print(f"learned tokenizers: collecting {n_img} images and {n_aud} clips (RAM only), {steps} training steps each")
    imgs = _img_tensor(collect_rows(("image",), {"image": n_img}, 1, PARTS, WORKERS))
    n_hold = max(16, len(imgs) // 10)
    tr, hold = imgs[:-n_hold], imgs[-n_hold:]
    print(f"  images: {len(tr)} train / {len(hold)} held out ({time.time() - t0:.0f}s)")
    base = ImageCodec(SPEC, dev)                                                        # k-means on the same data = the old codec
    base.fit([sample_points(base.features(tr[i:i + 64]), 200_000 // (len(tr) // 64 + 1)) for i in range(0, len(tr), 64)])
    ps_base = eval_image_tok(base, hold); del base
    ci = codecs.image
    train_vq(ci.net, lambda: tr[torch.randint(len(tr), (32,), device=tr.device)], _vq_loss(ci.net, LAYOUTS["image"].K, _img_pair), steps, 5e-4, "image")
    ci.mark_trained()
    ps_new = eval_image_tok(ci, hold)
    ppl = code_perplexity(ci, lambda: (ci.encode(hold[i:i + 32]) for i in range(0, len(hold), 32)))
    print(f"  image PSNR by level (dB): k-means {[round(p, 2) for p in ps_base]} | learned {[round(p, 2) for p in ps_new]}")
    print(f"  learned image code perplexity per level (256 = all codes used): {[round(p) for p in ppl]}")
    del imgs, tr, hold
    # ---------------- audio ----------------
    rows = collect_rows(("audio",), {"audio": n_aud}, 1, PARTS, WORKERS)
    wav = torch.from_numpy(np.stack([r["wav"] for r in rows])).to(dev); del rows
    n_hold = max(16, len(wav) // 10)
    trw, holdw = wav[:-n_hold], wav[-n_hold:]
    chunks = lambda w: [w[i:i + 32].float() / 32768 for i in range(0, len(w), 32)]
    ca = codecs.audio
    ca.set_stats(chunks(trw))
    base = AudioCodec(SPEC, dev); base.set_stats(chunks(trw))
    base.fit([sample_points(base.features(w), 200_000 // (len(trw) // 32 + 1)) for w in chunks(trw)])
    rm_base = eval_audio_tok(base, chunks(holdw)); del base
    def a_batch():
        with torch.no_grad():
            return ca.norm_mel(trw[torch.randint(len(trw), (16,), device=trw.device)].float() / 32768)
    train_vq(ca.net, a_batch, _vq_loss(ca.net, K, lambda x: (x, x)), steps, 5e-4, "audio")
    ca.mark_trained()
    rm_new = eval_audio_tok(ca, chunks(holdw))
    print(f"  audio log-mel RMSE by level: k-means {[round(p, 3) for p in rm_base]} | learned {[round(p, 3) for p in rm_new]}")
    del wav, trw, holdw
    gc.collect(); torch.cuda.empty_cache() if IS_CUDA else None
    print(f"learned tokenizers ready ({time.time() - t0:.0f}s)")
    # ---------------- gate ----------------
    bad = []
    if ps_new[-1] < ps_base[-1] + TOK_GATE_DB:
        bad.append(f"image finest PSNR {ps_new[-1]:.2f} dB is not {TOK_GATE_DB} dB above k-means {ps_base[-1]:.2f} dB")
    if rm_new[-1] > 0.95 * rm_base[-1]:
        bad.append(f"audio finest log-mel RMSE {rm_new[-1]:.3f} is not 5% below k-means {rm_base[-1]:.3f}")
    if bad and TOK_GATE_DB > -90:
        raise RuntimeError("tokenizer gate failed: " + "; ".join(bad) + ". Train longer (TOK_STEPS / TOK_IMAGES) or set TOK_GATE_DB=-99 to continue anyway.")
    if bad:
        print("  !! gate would have failed (ignored): " + "; ".join(bad))
    if cache:                                                                           # only reached when the gate passed (or was switched off)
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        torch.save(dict(key=key, codecs=codecs.state()), cache + ".tmp")
        os.replace(cache + ".tmp", cache)
        print(f"  tokenizers saved to {cache} ({os.path.getsize(cache) / 2**20:.0f} MB): later runs skip training")


def _img_pair(u8):
    x = u8.float() / 127.5 - 1.0
    if random.random() < 0.5:
        x = x.flip(-1)
    return x, x
