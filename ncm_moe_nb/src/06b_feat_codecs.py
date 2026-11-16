# %% [markdown]
# ## 3b. Feature codecs (`TOKENIZER="feat"`): frozen pretrained features, tree-quantised
#
# The k-means and learned-VQ codecs reconstruct pixels / mel frames. These codecs instead take the codes from **frozen
# pretrained features**, so the coarse rungs carry concept, not a blur (docs/codecs.md in `omni/`):
#
# | modality | features | levels | decode |
# |---|---|---|---|
# | image | DINOv2 patch tokens (224 px -> 16x16 grid) | the grid, 2x2 average-pooled per level | prototype *features* only: **no pixel decoder**, so no image generation / PSNR |
# | audio | DAC encoder latents (44.1 kHz), pooled to 512 frames | the frame rate pyramid | DAC decoder from prototype latents (a real waveform) |
# | text | `TEXT_CODEC`: `km` data-fit byte-embedding ladder (default) / `entropy` word-aligned spans / `hash` | spans | finest level = the exact byte |
#
# **Quantisation (`FEAT_QUANT`).** `tree`: one k-means codebook of 256 leaves on the finest features, a balanced binary tree built
# post hoc over it (Tree-VQ style), and level *k* = the ancestor of the leaf at `d_min + (K-1-k)` merges up, so coarse levels
# use fewer codes. `perlevel`: an independent 256-code k-means per level (the old style).
# Both pool the *features* to each level's length first, then quantise, so a level is a coarsening of the same feature map.
# Two fixes against the original `omni` implementation: depths are counted as merges up from the leaf everywhere (the
# original mixed that with depth from the root and could not build a tree for >256 leaves), and features are **not**
# centred per sample (that removed the content, and made a one-slot level constant).

# %%
import hashlib

FEAT_QUANT = _knob("FEAT_QUANT", "tree")           # "tree" | "perlevel"
TEXT_CODEC = _knob("TEXT_CODEC", "km")             # "km" | "entropy" | "hash" | "vq"   (only with TOKENIZER="feat"; "vq" = learned multiscale residual VQ-VAE, the NCM paper's text tokenizer)
TEXT_VQ_STEPS = _knob("TEXT_VQ_STEPS", 3000)       # TEXT_CODEC="vq": training steps of the text residual quantiser
TEXT_VQ_ROWS = _knob("TEXT_VQ_ROWS", 40000)        # ... and the 256-byte text rows it trains on (10% held out)
DINO_MODEL = _knob("DINO_MODEL", "facebook/dinov2-base")
DAC_MODEL = _knob("DAC_MODEL", "16khz")            # 16 kHz: about 2.6x cheaper to encode than 44.1 kHz (measured); the T1 run used it
FEAT_IMAGES = _knob("FEAT_IMAGES", 6000)           # images / clips / text rows streamed once to fit the codebooks (RAM only)
FEAT_AUDIO = _knob("FEAT_AUDIO", 2000)
FEAT_TEXT = _knob("FEAT_TEXT", 8000)
FEAT_AUDIO_BATCH = _knob("FEAT_AUDIO_BATCH", 8)    # DAC runs at 44.1 kHz in fp32: a batch of 32 clips of 8 s needs many GB on a small GPU
IMAGE_CODEC = _knob("IMAGE_CODEC", "dino")         # "dino" = frozen DINOv2 features (no pixel decoder, 16x16 grid) | "vq" = the learned multi-scale residual VQ-VAE of section 4b (pixel decoder; any IMAGE_RES)
VQ_IMAGES = _knob("VQ_IMAGES", 8192)               # IMAGE_CODEC="vq": images held in RAM to train it (10% held out for its PSNR)
VQ_STEPS = _knob("VQ_STEPS", 3000)                 # ... and its training steps (3000 steps took ~5.5 min on an RTX PRO 6000 at the 32x32 grid; 4000 gave the same PSNR within 0.2 dB)
AUDIO_VQ_STEPS = _knob("AUDIO_VQ_STEPS", 4000)     # AUDIO_RES="v2": training steps of the residual quantiser on the DAC latents
assert AUDIO_RES == "v1" or TOKENIZER == "feat", "AUDIO_RES='v2' (8 residual levels on the DAC latents) is built into the feature codecs: set TOKENIZER='feat'"
assert IMAGE_RES == "v1" or (TOKENIZER == "vq" or (TOKENIZER == "feat" and IMAGE_CODEC == "vq")), \
    "IMAGE_RES='v2' (32x32 grid, two codebooks) needs a learned image codec: set TOKENIZER='feat' with IMAGE_CODEC='vq' (or TOKENIZER='vq')"

if TOKENIZER == "feat" and INSTALL:
    for _m, _p in (("transformers", "transformers"), ("dac", "descript-audio-codec")):
        try:
            __import__(_m)
        except ImportError:
            print(f"installing {_p} (feature codecs) ...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=False)


class TreeHierarchy:
    """Balanced binary hierarchy over a flat codebook, built post hoc (Tree-VQ style, arXiv:2609.03641).

    Recursive median split along each node's first principal direction, so a coarse node is a prototype (the mean of the
    leaves under it) and not a bucket of unrelated vectors. "Climb c" = c merges up from the leaf: climb 0 is the leaf,
    climb `height` is the root."""

    def __init__(self, vectors):
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or len(X) < 2 or not np.all(np.isfinite(X)):
            raise ValueError(f"tree needs a finite 2-D codebook of >= 2 vectors, got {X.shape}")
        self.X, self.V = X, len(X)
        self._vec, self._parent = [], []
        self._leaf = np.full(self.V, -1, dtype=np.int64)
        self._build(np.arange(self.V))
        self._parent = np.asarray(self._parent, dtype=np.int64)
        assert (self._leaf >= 0).all()
        self._anc = {}
        c = 0
        while len(np.unique(self.climbed(c))) > 1:
            c += 1
        self.height = c

    @staticmethod
    def _pc1(X, iters=8):
        Xc = X - X.mean(0, keepdims=True)
        v = np.ones(Xc.shape[1], dtype=np.float32)
        for _ in range(iters):
            nv = Xc.T @ (Xc @ v)
            n = float(np.linalg.norm(nv))
            if n < 1e-8:
                return np.linspace(1.0, 2.0, Xc.shape[1], dtype=np.float32)
            v = nv / n
        return v

    def _build(self, idx):
        node = len(self._vec)
        self._vec.append(self.X[idx].mean(0))
        self._parent.append(-1)
        if idx.size == 1:
            self._leaf[int(idx[0])] = node
            return node
        order = idx[np.argsort((self.X[idx] - self._vec[node]) @ self._pc1(self.X[idx]), kind="stable")]
        half = idx.size // 2
        l, r = self._build(order[:half]), self._build(order[half:])
        self._parent[l] = self._parent[r] = node
        return node

    def climbed(self, c):
        """[V] node id of every leaf's ancestor after c merges up (a leaf already at the root stays there)."""
        if c not in self._anc:
            out = self._leaf.copy()
            for _ in range(int(c)):
                par = self._parent[out]
                out = np.where(par >= 0, par, out)
            self._anc[c] = out
        return self._anc[c]

    def level_map(self, c):
        """[V] compact code 1..n of every leaf at climb c, and n."""
        anc = self.climbed(c)
        uniq = np.unique(anc)
        lut = np.zeros(len(self._vec), dtype=np.int32)
        lut[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
        return lut[anc], len(uniq)

    def plan(self, K, cap):
        """Climbs per level (coarse -> fine): the finest level is the smallest climb whose node count fits `cap`."""
        d_min = 0
        while len(np.unique(self.climbed(d_min))) > cap and d_min < self.height:
            d_min += 1
        climbs = [min(d_min + (K - 1 - k), self.height) for k in range(K)]
        note = f"tree: {self.V} leaves, height {self.height}, finest level at climb {d_min}"
        if d_min + K - 1 > self.height:
            note += f" | !! tree shallower than K={K}: coarse levels share a partition (climbs {climbs})"
        return climbs, note


class FeatureLadderCodec:
    """Frozen features -> K nested levels of codes 1..256. Subclasses implement `extract` (raw finest feature map)."""
    decodable = False

    def __init__(self, spec, device, mod, nd, mode):
        self.spec, self.device, self.mod, self.nd, self.mode = spec, device, mod, nd, mode
        self.lens = spec.lens(mod)
        self.trained, self.cb, self.protos, self.note = False, None, None, ""
        self.mean, self.std = 0.0, 1.0                     # the shard fingerprint reads these on the audio codec

    def extract(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def features(self, x):
        """list K of raw pooled features [B, n_k, C]; level k = the finest map average-pooled to n_k slots."""
        f = self.extract(x)
        out = []
        for n in self.lens:
            p = F.adaptive_avg_pool2d(f, math.isqrt(n)).flatten(2) if self.nd == 2 else F.adaptive_avg_pool1d(f, n)
            out.append(p.transpose(1, 2).contiguous())
        return out

    @staticmethod
    def _nearest(xn, cb, chunk=65536):
        return torch.cat([(xn[i:i + chunk] @ cb.T).argmax(1) for i in range(0, len(xn), chunk)])

    @staticmethod
    def _protos(raw, a, n):
        s = torch.zeros(n, raw.shape[1], device=raw.device).index_add_(0, a, raw)
        c = torch.bincount(a, minlength=n).float()
        return s / c.clamp(min=1)[:, None], c

    @torch.no_grad()
    def fit(self, batches, sample=96):
        """batches: iterable of the codec's raw inputs (on the device). Keeps ~`sample` finest tokens per input (fp16)."""
        K, tree = self.spec.K, self.mode == "tree"
        pts = [[] for _ in range(K)]
        t0, nb = time.time(), len(batches)
        for bi, x in enumerate(batches):
            if bi == 0 or (bi + 1) % 25 == 0 or bi + 1 == nb:
                el = time.time() - t0
                print(f"      fit {self.mod}: batch {bi + 1}/{nb} | {el:.0f}s elapsed" + (f" | ~{el / bi * (nb - bi) / 60:.1f} min left" if bi else ""), flush=True)
            for k, f in enumerate(self.features(x)):
                if tree and k < K - 1:
                    continue
                f = f.reshape(-1, f.shape[-1])
                cap = sample * x.shape[0] if k == K - 1 else f.shape[0]
                if f.shape[0] > cap:
                    f = f[torch.randperm(f.shape[0], device=f.device)[:cap]]
                pts[k].append(f.half())
        unit = lambda a: F.normalize(a.float(), dim=-1)
        if tree:
            raw = torch.cat(pts[-1]).float()
            self.leaves = F.normalize(kmeans(unit(raw), 256, seed=0), dim=-1)
            self.tree = TreeHierarchy(self.leaves.cpu().numpy())
            climbs, self.note = self.tree.plan(K, 256)
            maps = [self.tree.level_map(c) for c in climbs]
            self.lut = [torch.from_numpy(m).long().to(self.device) for m, _ in maps]          # [V] -> 1..n_k
            a = self._nearest(unit(raw), self.leaves)
            leaf_mean, leaf_cnt = self._protos(raw, a, len(self.leaves))
            leaf_sum = leaf_mean * leaf_cnt[:, None]
            self.protos = []
            for k in range(K):
                idx, n = self.lut[k] - 1, maps[k][1]
                s = torch.zeros(n, raw.shape[1], device=raw.device).index_add_(0, idx, leaf_sum)
                c = torch.zeros(n, device=raw.device).index_add_(0, idx, leaf_cnt)
                self.protos.append(s / c.clamp(min=1)[:, None])
            self.cb = [F.normalize(p, dim=-1) for p in self.protos]
            self.note += f" | nodes per level {[int(m[1]) for m in maps]}"
        else:
            self.cbs, self.protos = [], []
            for k in range(K):
                raw = torch.cat(pts[k]).float()
                cb = F.normalize(kmeans(unit(raw), 256, seed=k), dim=-1)
                p, _ = self._protos(raw, self._nearest(unit(raw), cb), 256)
                self.cbs.append(cb); self.protos.append(p)
            self.cb = self.cbs
            self.note = "per-level k-means, 256 codes each"
        self.trained = True

    @torch.no_grad()
    def encode(self, x):
        out = []
        for k, f in enumerate(self.features(x)):
            B, n, C = f.shape
            xn = F.normalize(f.reshape(-1, C), dim=-1)
            if self.mode == "tree":
                out.append(self.lut[k][self._nearest(xn, self.leaves)].view(B, n))
            else:
                out.append((self._nearest(xn, self.cbs[k]) + 1).view(B, n))
        return out

    def reconstruct(self, codes, level):
        """prototype feature of every code of one level: [B, n_k, C] (what the codes keep of the features)."""
        return self.protos[level][(codes - 1).clamp(0, self.protos[level].shape[0] - 1)]     # the model may sample a code this level never uses

    def state(self):
        if not self.trained:
            return None
        d = dict(kind="feat", mode=self.mode, protos=[p.cpu() for p in self.protos], note=self.note)
        if self.mode == "tree":
            d.update(leaves=self.leaves.cpu(), lut=[l.cpu() for l in self.lut])
        else:
            d.update(cbs=[c.cpu() for c in self.cbs])
        return d

    def load(self, st):
        assert st.get("kind") == "feat" and st["mode"] == self.mode, \
            f"codec state is {st.get('kind')}/{st.get('mode')} but TOKENIZER='feat' FEAT_QUANT='{self.mode}': use a new CKPT_DIR"
        self.protos = [p.to(self.device) for p in st["protos"]]
        if self.mode == "tree":
            self.leaves, self.lut = st["leaves"].to(self.device), [l.to(self.device) for l in st["lut"]]
            self.cb = [F.normalize(p, dim=-1) for p in self.protos]
        else:
            self.cbs = [c.to(self.device) for c in st["cbs"]]
            self.cb = self.cbs
        self.note, self.trained = st.get("note", ""), True


class DinoImageCodec(FeatureLadderCodec):
    """DINOv2 patch tokens of the image resized to 224 px: a 16x16 grid, the same grid as the finest image level."""
    decodable = False

    def __init__(self, spec, device, mode):
        super().__init__(spec, device, "image", 2, mode)
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()
        self.n_reg = int(getattr(self.net.config, "num_register_tokens", 0))
        self.m = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.s = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(self, img):
        hs = []
        for i in range(0, img.shape[0], 32):                                 # chunks bound the activation memory next to a running training step
            x = F.interpolate(img[i:i + 32].float() / 255.0, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(IS_CUDA)):
                hs.append(self.net(pixel_values=(x - self.m) / self.s).last_hidden_state[:, 1 + self.n_reg:].float())   # drop CLS (+ register tokens)
        h = torch.cat(hs)
        g = math.isqrt(h.shape[1])
        assert g * g == h.shape[1], f"DINO returned {h.shape[1]} patch tokens: not a square grid"
        return h.transpose(1, 2).reshape(h.shape[0], -1, g, g)

    def decode(self, codes, level=None):
        raise NotImplementedError("DINO feature codes have no pixel decoder: image generation / PSNR are not available with this codec")


class DacAudioFeatureCodec(FeatureLadderCodec):
    """DAC (44.1 kHz) encoder latents, pooled to the notebook's frame count; decoded by the DAC decoder from prototype latents."""
    decodable = True
    SR = 16000 if DAC_MODEL == "16khz" else 44100

    def __init__(self, spec, device, mode):
        super().__init__(spec, device, "audio", 1, mode)
        import dac
        self.net = dac.DAC.load(dac.utils.download(model_type=DAC_MODEL)).to(device).eval()
        self.hop = int(getattr(self.net, "hop_length", 512))
        self.n_in = round(spec.wave_len * self.SR / spec.sr)
        self.n_in += (-self.n_in) % self.hop
        self.T = self.n_in // self.hop

    @torch.no_grad()
    def extract(self, wave):
        outs = []
        for i in range(0, wave.shape[0], FEAT_AUDIO_BATCH):                  # a whole micro-batch at 44.1 kHz in fp32 needs tens of GB: encode in small chunks
            x = wave[i:i + FEAT_AUDIO_BATCH].float()
            x = F.interpolate(x[:, None], size=round(x.shape[1] * self.SR / self.spec.sr), mode="linear", align_corners=False)
            z = self.net.encoder(F.pad(x, (0, self.n_in - x.shape[-1])))     # [b, 1024, T]
            outs.append(F.adaptive_avg_pool1d(z.float(), self.spec.audio))   # frames at the notebook's rate
        return torch.cat(outs)

    @torch.no_grad()
    def decode(self, codes, level=None, **_):
        level = self.spec.K - 1 if level is None else level
        if isinstance(codes, (list, tuple)):
            codes = codes[level]
        return self._wave(self.reconstruct(codes, level).transpose(1, 2))    # [B, C, n_k]

    @torch.no_grad()
    def _wave(self, z):
        """DAC latents [B, C, frames] -> waveform [B, wave_len] at the notebook's sample rate."""
        z = F.interpolate(z, size=self.T, mode="linear", align_corners=False)
        w = self.net.decoder(z)                                              # [B, 1, n_in] at 44.1 kHz
        w = F.interpolate(w, size=round(w.shape[-1] * self.spec.sr / self.SR), mode="linear", align_corners=False)[:, 0]
        return w[:, :self.spec.wave_len]


class AudioAdapterVQ(nn.Module):
    """Small conv encoder / decoder around a multi-scale residual quantiser (MSRVQ of section 4b): DAC latents -> d-dim -> 8-bit codes per level -> back."""

    def __init__(self, c_in, spec, d=64, ch=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(c_in, ch, 5, padding=2), _Res1d(ch, 1), _Res1d(ch, 2), _gn(ch), nn.SiLU(), nn.Conv1d(ch, d, 1))
        self.dec = nn.Sequential(nn.Conv1d(d, ch, 5, padding=2), _Res1d(ch, 1), _Res1d(ch, 2), _gn(ch), nn.SiLU(), nn.Conv1d(ch, c_in, 5, padding=2))
        self.rvq = MSRVQ(spec.lens("audio"), d, 1)


class DacResidualAudioCodec(DacAudioFeatureCodec):
    """AUDIO_RES="v2": DAC encoder latents (frozen, as in DacAudioFeatureCodec) quantised by a LEARNED residual quantiser with 8-bit codebooks.

    The ladder is the usual 5 levels plus `audio_extra` residual levels at the finest frame rate, each coding what the levels before it
    missed, so decoding level k needs the codes of all levels <= k. The frozen DAC decoder turns the rebuilt latents into audio."""
    decodable = True

    def __init__(self, spec, device, mode):
        super().__init__(spec, device, mode)
        self.K_aud = len(spec.lens("audio"))
        self.adapter = AudioAdapterVQ(int(getattr(self.net, "latent_dim", 1024)), spec).to(device).eval()
        self.mu = self.sd = None

    def fit(self, batches):                                  # not under no_grad: the quantiser is trained here
        with torch.no_grad():
            zs = [self.extract(w).half() for w in batches]
            z = torch.cat(zs); del zs
            ref = z[:256].float()
            self.mu, self.sd = ref.mean((0, 2), keepdim=True), ref.std((0, 2), keepdim=True).clamp(min=1e-4)
        n_hold = max(16, len(z) // 10)
        tr, hold = z[:-n_hold], z[-n_hold:]
        print(f"      audio residual quantiser: {len(tr)} train / {len(hold)} held-out clips, {AUDIO_VQ_STEPS} steps, ladder {self.lens}", flush=True)
        batch = lambda: (tr[torch.randint(len(tr), (16,), device=tr.device)].float() - self.mu) / self.sd
        train_vq(self.adapter, batch, _vq_loss(self.adapter, self.K_aud, lambda x: (x, x)), AUDIO_VQ_STEPS, 5e-4, "audio")
        self.cb, self.trained = [q.embed for q in self.adapter.rvq.q], True
        with torch.no_grad():
            x = (hold[:32].float() - self.mu) / self.sd
            with _amp_bf16():
                codes = self.adapter.rvq(self.adapter.enc(x).float())[1]
            r2 = []
            for k in range(self.K_aud):
                with _amp_bf16():
                    y = self.adapter.dec(self.adapter.rvq.from_codes(codes, k))
                r2.append(1 - float(((y.float() - x) ** 2).mean() / x.var()))
        self.note = f"latent R^2 by level {[round(r, 3) for r in r2]}"
        del z, tr, hold

    @torch.no_grad()
    def encode(self, wave):
        x = (self.extract(wave) - self.mu) / self.sd
        with _amp_bf16():
            z = self.adapter.enc(x)
        return self.adapter.rvq(z.float())[1]

    @torch.no_grad()
    def decode(self, codes, level=None, **_):
        level = self.K_aud - 1 if level is None else level
        with _amp_bf16():
            y = self.adapter.dec(self.adapter.rvq.from_codes(codes, level))
        return self._wave(y.float() * self.sd + self.mu)

    def state(self):
        if not self.trained:
            return None
        return dict(kind="feat_res", sd={k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}, mu=self.mu.cpu(), std=self.sd.cpu(), note=self.note)

    def load(self, st):
        assert st.get("kind") == "feat_res", "this checkpoint holds another audio codec: use a new CKPT_DIR"
        self.adapter.load_state_dict(st["sd"]); self.adapter.eval()
        self.mu, self.sd = st["mu"].to(self.device), st["std"].to(self.device)
        self.cb, self.trained, self.note = [q.embed for q in self.adapter.rvq.q], True, st.get("note", "")


# ---- text ----
class TextKMCodec:
    """Data-fit ladder: byte embeddings from the SVD of the byte-PPMI, span features = their average, k-means per level.
    The finest level is the exact byte, so decoding is lossless."""
    D = 16
    decode = staticmethod(TextCodec.decode)

    def __init__(self, spec, device):
        self.spec, self.device, self.emb, self.cb, self.trained = spec, device, None, None, False

    @torch.no_grad()
    def _fit_emb(self, b):
        b = b.long()
        C = torch.zeros(256 * 256, device=b.device)
        for d in (1, 2, 3, 4):
            C += torch.bincount(b[:, :-d].reshape(-1) * 256 + b[:, d:].reshape(-1), minlength=256 * 256).float()
        C = C.view(256, 256) + C.view(256, 256).T
        tot = C.sum()
        pr = C.sum(1, keepdim=True) / tot
        U, S, _ = torch.linalg.svd(torch.log((C / tot + 1e-12) / (pr @ pr.T + 1e-12)).clamp(min=0))
        e = U[:, :self.D] * S[:self.D].sqrt()
        return e / e.std()

    def features(self, b):
        K, R = self.spec.K, self.spec.R_of("text")
        e = self.emb[b.long()].transpose(1, 2)
        return [F.avg_pool1d(e, R ** (K - 1 - j)).transpose(1, 2).contiguous() for j in range(K - 1)]

    @torch.no_grad()
    def fit(self, b):
        self.emb = self._fit_emb(b)
        self.cb = [kmeans(f.reshape(-1, self.D), 256, seed=j) for j, f in enumerate(self.features(b))]
        self.trained = True

    @torch.no_grad()
    def encode(self, b):
        codes = [vq(f, self.cb[j]) for j, f in enumerate(self.features(b))]
        codes.append(b.long() + 1)
        return codes

    def state(self):
        return None if self.cb is None else dict(kind="text_km", emb=self.emb.cpu(), cb=[c.cpu() for c in self.cb])

    def load(self, st):
        assert st.get("kind") == "text_km", "the checkpoint holds another text codec: use a new CKPT_DIR"
        self.emb, self.cb, self.trained = st["emb"].to(self.device), [c.to(self.device) for c in st["cb"]], True


class TextEntropyCodec:
    """omni's word-aligned span codec: coarse levels cut at word starts, spent first where the next 4-gram is novel, each span
    coded by a hash of its bytes. Deterministic and corpus-free. Slower than the others (a python loop per row)."""
    NGRAM = 4
    decode = staticmethod(TextCodec.decode)

    def __init__(self, spec, device, seed=1):
        self.spec, self.device, self.seed, self.trained = spec, device, int(seed), True

    def _repeat(self, raw):
        seen, out = {}, np.zeros(len(raw), dtype=np.int32)
        for i in range(len(raw)):
            w = bytes(raw[i:i + self.NGRAM])
            out[i] = seen.get(w, 0)
            seen[w] = out[i] + 1
        return out

    @staticmethod
    def _candidates(raw):
        n = len(raw)
        starts = [i for i in range(1, n) if raw[i] not in b" \t\n\r" and raw[i - 1] in b" \t\n\r"]
        return starts or list(range(1, n))

    def _edges(self, raw, l, cand, rep):
        n = len(raw)
        if l <= 1:
            return [0, n]
        if n <= 1:
            return [0] * (l + 1)
        m = l - 1
        if len(cand) > m:
            order = np.lexsort((np.asarray(cand), rep[cand]))
            cand = sorted(int(cand[j]) for j in order[:m])
        elif len(cand) < m:
            fill = np.linspace(1, n - 1, m).round().astype(int)
            cand = sorted(set(cand) | {int(p) for p in fill if 1 <= p < n})[:m]
        return [0] + list(cand) + [n]

    def _codes(self, raw, l, cand, rep):
        edges = self._edges(raw, l, cand, rep)
        if len(edges) - 1 != l:
            edges = [round(i * len(raw) / l) for i in range(l + 1)]
        out = np.empty(l, dtype=np.int64)
        for i in range(l):
            h = hashlib.blake2b(bytes(raw[edges[i]:edges[i + 1]]), digest_size=8, salt=self.seed.to_bytes(8, "little")).digest()
            out[i] = int.from_bytes(h, "little") % 256 + 1
        return out

    @torch.no_grad()
    def encode(self, b):
        rows, lens, K = b.cpu().numpy(), self.spec.lens("text"), self.spec.K
        out = [np.zeros((len(rows), n), dtype=np.int64) for n in lens]
        for i, r in enumerate(rows):
            raw = r.tobytes().rstrip(b"\x00")
            cand, rep = self._candidates(raw), self._repeat(raw)
            for k in range(K - 1):
                out[k][i] = self._codes(raw, lens[k], cand, rep)
        out[K - 1] = rows.astype(np.int64) + 1
        return [torch.from_numpy(o).to(self.device) for o in out]

    def state(self):
        return None

    def load(self, st):
        pass


class TextVQNet(nn.Module):
    """Bytes [B, L] -> L/R latent slots -> residual 8-bit codes for every level above the finest -> byte logits [B, 256, L]."""

    def __init__(self, spec, d=64, ch=256):
        super().__init__()
        r = spec.R_of("text")
        self.emb = nn.Embedding(256, 128)
        self.enc = nn.Sequential(nn.Conv1d(128, ch, 5, padding=2), _Res1d(ch, 1), nn.Conv1d(ch, ch, r, stride=r), _Res1d(ch, 1), _Res1d(ch, 2),
                                 _gn(ch), nn.SiLU(), nn.Conv1d(ch, d, 1))
        self.dec = nn.Sequential(nn.Conv1d(d, ch, 5, padding=2), _Res1d(ch, 1), _Res1d(ch, 2), nn.Upsample(scale_factor=r, mode="nearest"),
                                 nn.Conv1d(ch, ch, 5, padding=2), _Res1d(ch, 1), _gn(ch), nn.SiLU(), nn.Conv1d(ch, 256, 1))
        self.rvq = MSRVQ(spec.lens("text")[:-1], d, 1)             # every level except the finest, which is the exact byte

    def encode(self, b):
        return self.enc(self.emb(b.long()).transpose(1, 2))


class TextVQCodec:
    """TEXT_CODEC="vq": the NCM paper's idea for text (a multiscale residual VQ-VAE, trained with nested dropout), adapted to this ladder.

    Levels 0..K-2 are learned residual 8-bit codes, trained so a decoder can rebuild the bytes from the codes of the first k levels (a random
    prefix is trained as well as the full stack). The finest level stays the exact byte, so decoding is lossless."""
    decode = staticmethod(TextCodec.decode)

    def __init__(self, spec, device):
        self.spec, self.device, self.trained, self.note = spec, device, False, ""
        self.net = TextVQNet(spec).to(device).eval()

    def fit(self, b):
        b = b.long()
        n_hold = max(64, len(b) // 10)
        tr, hold = b[:-n_hold], b[-n_hold:]
        net = self.net

        def loss_fn(x):
            z = net.encode(x).float()
            outs, _, commit = net.rvq(z)
            ce = lambda f: F.cross_entropy(net.dec(f).float(), x)
            full = ce(outs[-1])
            k = random.randrange(len(outs) - 1) if len(outs) > 1 else 0
            return full + 0.5 * ce(outs[k]) + 0.25 * commit, dict(ce=float(full), commit=float(commit))

        print(f"      text residual quantiser: {len(tr)} train / {len(hold)} held-out rows, {TEXT_VQ_STEPS} steps, learned levels {self.spec.lens('text')[:-1]} + exact byte", flush=True)
        train_vq(net, lambda: tr[torch.randint(len(tr), (64,), device=tr.device)], loss_fn, TEXT_VQ_STEPS, 5e-4, "text")
        self.trained = True
        with torch.no_grad():
            x = hold[:256]
            with _amp_bf16():
                codes = net.rvq(net.encode(x).float())[1]
            acc, pad = [], x == 0
            for k in range(len(codes)):
                with _amp_bf16():
                    pred = net.dec(net.rvq.from_codes(codes, k)).argmax(1)
                acc.append(float(((pred == x) & ~pad).sum() / (~pad).sum().clamp(min=1)))
        self.note = f"byte accuracy rebuilt from the codes of levels 0..k: {[round(a, 3) for a in acc]}"

    @torch.no_grad()
    def encode(self, b):
        with _amp_bf16():
            z = self.net.encode(b)
        return list(self.net.rvq(z.float())[1]) + [b.long() + 1]

    def state(self):
        return dict(kind="text_vq", sd={k: v.detach().cpu() for k, v in self.net.state_dict().items()}, note=self.note) if self.trained else None

    def load(self, st):
        assert st.get("kind") == "text_vq", "the checkpoint holds another text codec: use a new CKPT_DIR"
        self.net.load_state_dict(st["sd"]); self.net.eval()
        self.trained, self.note = True, st.get("note", "")


def make_text_codec(spec, device):
    if TOKENIZER != "feat" or TEXT_CODEC == "hash":
        return TextCodec(spec, device)
    return {"km": TextKMCodec, "entropy": TextEntropyCodec, "vq": TextVQCodec}[TEXT_CODEC](spec, device)


def fit_image_vq(t0):
    """IMAGE_CODEC="vq": train the multi-scale residual image VQ-VAE (section 4b) on streamed images; the DAC audio and text codecs are fitted next to it."""
    path = os.path.join(CKPT_DIR, "image_vq.pt") if CKPT_DIR else ""          # saved right after training: a later failure (audio fit) must not cost 5-8 min again
    key = repr((repr(SPEC), VQ_IMAGES, VQ_STEPS))
    if path and os.path.exists(path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st.get("key") == key:
            codecs.image.load(st["state"])
            print(f"  image VQ restored from {path} (trained earlier with the same settings: no retrain)")
            return
    imgs = _img_tensor(collect_rows(("image",), {"image": VQ_IMAGES}, 1, PARTS, WORKERS))
    n_hold = max(16, len(imgs) // 10)
    tr, hold = imgs[:-n_hold], imgs[-n_hold:]
    ci, K_img = codecs.image, LAYOUTS["image"].K
    print(f"  image VQ: {len(tr)} train / {len(hold)} held-out images, {VQ_STEPS} steps, ladder {LAYOUTS['image'].lens} ({time.time() - t0:.0f}s)", flush=True)
    train_vq(ci.net, lambda: tr[torch.randint(len(tr), (32,), device=tr.device)], _vq_loss(ci.net, K_img, _img_pair), VQ_STEPS, 5e-4, "image")
    ci.mark_trained()
    ps = eval_image_tok(ci, hold)
    ppl = code_perplexity(ci, lambda: (ci.encode(hold[i:i + 32]) for i in range(0, len(hold), 32)))
    print(f"  image PSNR by level (dB): {[round(p, 2) for p in ps]} | code perplexity per level (256 = all codes used): {[round(p) for p in ppl]}")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(key=key, state=ci.state()), path + ".tmp"); os.replace(path + ".tmp", path)
        print(f"  image VQ saved to {path}")
    del imgs, tr, hold


def fit_feature_codecs():
    """Stream a sample into RAM, fit the image / audio / text codecs on it, cache the result (no model compute yet)."""
    t0 = time.time()
    cache = _knob("TOK_CACHE", os.path.join(CKPT_DIR, "feat_codecs.pt") if CKPT_DIR else "")
    key = repr((repr(SPEC), TEXT_CODEC, FEAT_QUANT, FEAT_IMAGES, FEAT_AUDIO, FEAT_TEXT, DINO_MODEL, DAC_MODEL, IMAGE_CODEC, VQ_IMAGES, VQ_STEPS, AUDIO_RES, AUDIO_VQ_STEPS, TEXT_VQ_STEPS, TEXT_VQ_ROWS))
    if cache and os.path.exists(cache):
        st = torch.load(cache, map_location="cpu", weights_only=False)
        if st.get("key") == key:
            codecs.load(st["codecs"])
            print(f"feature codecs restored from {cache} (fitted earlier with the same settings: no refit)")
            return
        print(f"!! {cache} was fitted with other settings: refitting (and overwriting it)")
    print(f"feature codecs: {'training the image VQ on ' + str(VQ_IMAGES) if IMAGE_CODEC == 'vq' else 'streaming ' + str(FEAT_IMAGES)} images, {FEAT_AUDIO} clips, {FEAT_TEXT} text rows (RAM only)")
    if IMAGE_CODEC == "vq":
        fit_image_vq(t0)
    else:
        imgs = _img_tensor(collect_rows(("image",), {"image": FEAT_IMAGES}, 1, PARTS, WORKERS))
        codecs.image.fit([imgs[i:i + 64] for i in range(0, len(imgs), 64)])
        print(f"  image ({DINO_MODEL}, {FEAT_QUANT}): {codecs.image.note} ({time.time() - t0:.0f}s)")
        del imgs
    rows = collect_rows(("audio",), {"audio": FEAT_AUDIO}, 1, PARTS, WORKERS)
    wav = torch.from_numpy(np.stack([r["wav"] for r in rows])).to(dev); del rows
    codecs.audio.fit([wav[i:i + FEAT_AUDIO_BATCH].float() / 32768 for i in range(0, len(wav), FEAT_AUDIO_BATCH)])
    print(f"  audio (DAC {DAC_MODEL}, {FEAT_QUANT}): {codecs.audio.note} ({time.time() - t0:.0f}s)")
    del wav
    if hasattr(codecs.text, "fit"):
        n_text = TEXT_VQ_ROWS if TEXT_CODEC == "vq" else FEAT_TEXT
        rows = collect_rows(("text0",), {"text0": n_text}, 1, PARTS, WORKERS)
        codecs.text.fit(torch.from_numpy(np.stack([r["tgt"] for r in rows])).to(dev)); del rows
    print(f"  text ({type(codecs.text).__name__}) ready" + (f": {codecs.text.note}" if getattr(codecs.text, "note", "") else "") + f" ({time.time() - t0:.0f}s)")
    gc.collect(); torch.cuda.empty_cache() if IS_CUDA else None
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        torch.save(dict(key=key, codecs=codecs.state()), cache + ".tmp")
        os.replace(cache + ".tmp", cache)
        print(f"  codecs saved to {cache}: later runs skip the fit")
