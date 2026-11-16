# %% [markdown]
# ## 7d. Codec checks (before the model is built: no GPU time is spent on a bad tokenizer)
#
# Streams a small fresh sample of every modality through the **frozen codecs that were just fitted** and checks:
#
# 1. **Contract** (hard): every level has the ladder length, codes are in 1..256, no level is constant, no NaN, and the finest text
#    level decodes to the exact bytes. A failure here stops the run (`CODEC_GATE="stop"`).
# 2. **What the codes keep** (soft): image PSNR (pixel codecs) or feature R^2 (DINO), audio log-mel R^2, per level.
# 3. **How the codes are used** (soft): entropy, codes used and perplexity per level (a level near 0 bits or with few codes is wasted).
# 4. **Learnability** (soft): (a) the bits the parent level tells about each level (a first-order estimate, bias-controlled with a
#    shuffled control) and (b) a **probe**: a small conv net trained for a few hundred steps to predict level *k* from level *k-1*,
#    scored on held-out rows against the unigram. The probe's gain is a rough ceiling on what the big model can win at that level
#    from the previous one, so a level whose probe gain is ~0 will sit at the constant baseline however long the model trains.
#
# The probe results are kept in `PROBE` and compared with the trained model in the final "did it learn" cells.

# %%
CODEC_CHECK = _knob("CODEC_CHECK", True)
CODEC_GATE = _knob("CODEC_GATE", "stop")            # "stop" = raise on a hard failure | "warn" = print and continue
CHK_IMAGES, CHK_AUDIO, CHK_TEXT = _knob("CHK_IMAGES", 1200), _knob("CHK_AUDIO", 500), _knob("CHK_TEXT", 3000)
CHK_PROBE_STEPS = _knob("CHK_PROBE_STEPS", 300)
PROBE, CODEC_FLAGS, CODEC_REPORT = {}, [], {}


def _flag(sev, msg):
    CODEC_FLAGS.append((sev, msg))
    print(("  !! HARD: " if sev == "HARD" else "  ~ ") + msg, flush=True)


def _H(counts):
    p = counts.double(); p = p[p > 0]; p = p / p.sum()
    return float(-(p * p.log2()).sum())


def _cond_H(par, ch):                                        # H(child | context) in bits, both codes 1..256
    V = 257
    return _H(torch.bincount(par * V + ch, minlength=V * V)) - _H(torch.bincount(par, minlength=V))


def _parent_index(k, mod):                                   # parent slot (level k-1) of every slot at level k
    lens = SPEC.lens(mod)
    n, f = lens[k], math.isqrt(SPEC.R_of(mod))
    i = torch.arange(n, device=dev)
    if k and lens[k - 1] == n:                               # a residual level at the same resolution: the parent is the same slot
        return i
    if mod == "image":
        g = math.isqrt(n)
        return ((i // g) // f) * (g // f) + ((i % g) // f)
    return i // SPEC.R_of(mod)


def _upsample_parent(prev, n_k, mod):                        # [B, n_{k-1}] -> [B, n_k]: every slot sees its parent's code
    B, n_p = prev.shape
    r = n_k // n_p
    if mod == "image":
        g_p, f = math.isqrt(n_p), math.isqrt(r)
        return prev.view(B, g_p, g_p).repeat_interleave(f, 1).repeat_interleave(f, 2).reshape(B, n_k)
    return prev.repeat_interleave(r, 1)


class _Probe(nn.Module):
    """Small conv net: embed the parent's code at every slot (+ a learned position), 3 conv layers, predict the 257-way code."""

    def __init__(self, n, nd, d=64):
        super().__init__()
        self.nd = nd
        self.emb = nn.Embedding(257, d)
        self.pos = nn.Parameter(torch.zeros(1, d, *([math.isqrt(n)] * 2 if nd == 2 else [n])))
        conv, ks = (nn.Conv2d, 3) if nd == 2 else (nn.Conv1d, 5)
        self.net = nn.Sequential(conv(d, 128, ks, padding=ks // 2), nn.GELU(), conv(128, 128, ks, padding=ks // 2), nn.GELU(),
                                 conv(128, 128, ks, padding=ks // 2), nn.GELU(), conv(128, 257, 1))

    def forward(self, up):                                   # long [B, n] -> logits [B, 257, n]
        x = self.emb(up).transpose(1, 2)
        if self.nd == 2:
            g = math.isqrt(up.shape[1])
            x = x.reshape(up.shape[0], -1, g, g)
        return self.net(x + self.pos).flatten(2)


def _probe_level(codes, k, mod, steps):
    """(unigram bits, probe bits) per code on held-out rows for level k predicted from level k-1."""
    tgt = codes[k]
    N, n = tgt.shape
    n_tr = int(N * 0.8)
    up = _upsample_parent(codes[k - 1], n, mod)
    p = _Probe(n, 2 if mod == "image" else 1).to(dev)
    opt = torch.optim.AdamW(p.parameters(), lr=2e-3, weight_decay=0.01)
    with torch.enable_grad():                                # the caller runs under no_grad
        for _ in range(steps):
            idx = torch.randint(n_tr, (min(64, n_tr),), device=dev)
            loss = F.cross_entropy(p(up[idx]), tgt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        ce = F.cross_entropy(p(up[n_tr:]), tgt[n_tr:]).item()
        cnt = torch.bincount(tgt[:n_tr].flatten(), minlength=257).float() + 1
        uni = float(-(cnt / cnt.sum()).log()[tgt[n_tr:].flatten()].mean())
    return uni / math.log(2), ce / math.log(2)


def _analyse(name, mod, codes):
    """contract + usage + learnability for one modality; codes = K tensors [N, n_k] (long, 1..256)."""
    lens = SPEC.lens(mod)
    print(f"\n{name}: level | slots | codes used | entropy (bits, max 8) | perplexity | parent info (bits) | probe: unigram -> probe bits, gain")
    if [c.shape[1] for c in codes] != list(lens):
        _flag("HARD", f"{name}: level lengths {[c.shape[1] for c in codes]} != ladder {lens}")
    rows = []
    for k, c in enumerate(codes):
        cnt = torch.bincount(c.flatten(), minlength=257)
        h, used = _H(cnt), int((cnt[1:] > 0).sum())
        if int(c.min()) < 1 or int(c.max()) > 256:
            _flag("HARD", f"{name} L{k}: codes outside 1..256 (min {int(c.min())}, max {int(c.max())})")
        if used < 2:
            _flag("HARD", f"{name} L{k}: every code is identical (a constant level carries no information)")
        elif lens[k] >= 16 and used < 16:
            _flag("SOFT", f"{name} L{k}: only {used} codes used across {lens[k]} slots (wasted capacity)")
        line = f"   L{k}  {c.shape[1]:4d}   {used:3d}   {h:5.2f}   {2 ** h:6.1f}"
        if k == 0:
            print(line); rows.append(dict(level=k, entropy=h, used=used)); continue
        par, ch = codes[k - 1][:, _parent_index(k, mod)].flatten(), c.flatten()
        gain_par = _cond_H(par, ch[torch.randperm(len(ch), device=ch.device)]) - _cond_H(par, ch)
        uni, pb = _probe_level(codes, k, mod, CHK_PROBE_STEPS)
        PROBE[(mod, k)] = dict(uni_bits=uni, probe_bits=pb, gain_bits=uni - pb, parent_bits=gain_par, entropy=h, used=used)
        print(f"{line}   {gain_par:5.2f}   {uni:5.2f} -> {pb:5.2f}  ({uni - pb:+.2f})")
        rows.append(dict(level=k, entropy=h, used=used, parent=gain_par, probe_gain=uni - pb))
    if rows[-1]["entropy"] < 2.0:
        _flag("SOFT", f"{name}: the finest level has only {rows[-1]['entropy']:.2f} bits of entropy")
    if len(rows) > 1 and rows[-1].get("probe_gain", 1) < 0.05:
        _flag("SOFT", f"{name}: the finest level gains only {rows[-1]['probe_gain']:+.3f} bits from its parent level (little learnable structure)")
    return rows


def _fmt_levels(vals, fmt):
    return " ".join(f"L{k}=" + fmt.format(v) for k, v in enumerate(vals))


@torch.no_grad()
def run_codec_checks():
    K = SPEC.K
    print(f"codecs: image {type(codecs.image).__name__} | audio {type(codecs.audio).__name__} | text {type(codecs.text).__name__} "
          f"| TOKENIZER={TOKENIZER}" + (f" FEAT_QUANT={FEAT_QUANT} TEXT_CODEC={TEXT_CODEC}" if TOKENIZER == "feat" else ""))
    # ---------------- image ----------------
    try:
        imgs = _img_tensor(collect_rows(("image",), {"image": CHK_IMAGES}, 1, PARTS, WORKERS))
        Ki = LAYOUTS["image"].K                              # images may have one more level than SPEC.K (IMAGE_RES="v2")
        acc, num, den, kind = [[] for _ in range(Ki)], [0.0] * Ki, [0.0] * Ki, None
        for i in range(0, len(imgs), 64):
            x = imgs[i:i + 64]
            cl = codecs.image.encode(x)
            for k in range(Ki):
                acc[k].append(cl[k])
            if getattr(codecs.image, "decodable", True):
                kind = "psnr"
                for k in range(Ki):
                    num[k] += float(((codecs.image.decode(cl, level=k).float() - x.float()) ** 2).sum()); den[k] += x.numel()
            else:
                kind, feats = "r2", codecs.image.features(x)
                for k in range(Ki):
                    f = feats[k].reshape(-1, feats[k].shape[-1]); p = codecs.image.reconstruct(cl[k], k).reshape(f.shape)
                    num[k] += float(((f - p) ** 2).sum()); den[k] += float(((f - f.mean(0)) ** 2).sum())
        if kind == "psnr":
            rec = [10 * math.log10(255 ** 2 / max(num[k] / den[k], 1e-9)) for k in range(Ki)]
            print(f"IMAGE: {len(imgs)} images | PSNR from the codes of levels 0..k (dB): " + _fmt_levels(rec, "{:.1f}"))
        else:
            rec = [1 - num[k] / max(den[k], 1e-9) for k in range(Ki)]
            print(f"IMAGE: {len(imgs)} images | DINO-feature R^2 of the prototype features per level (1.0 = perfect): " + _fmt_levels(rec, "{:.2f}"))
            if rec[-1] < 0.3:
                _flag("SOFT", f"image finest-level feature R^2 {rec[-1]:.2f} is low: the 256 codes keep little of the features")
        if not all(math.isfinite(r) for r in rec):
            _flag("HARD", "image reconstruction metric is not finite")
        CODEC_REPORT["image"] = dict(kind=kind, recon=rec, rows=_analyse("IMAGE", "image", [torch.cat(a) for a in acc]))
        del imgs
    except Exception as e:
        _flag("HARD", f"image codec failed: {type(e).__name__}: {str(e)[:200]}")
    # ---------------- audio ----------------
    try:
        rows = collect_rows(("audio",), {"audio": CHK_AUDIO}, 1, PARTS, WORKERS)
        wav = torch.from_numpy(np.stack([r["wav"] for r in rows])).to(dev); del rows
        helper = AudioCodec(SPEC, dev)                       # only its STFT / mel filterbank: the same mel space for every audio codec
        Ka = LAYOUTS["audio"].K                              # audio may have residual levels beyond SPEC.K (AUDIO_RES="v2")
        acc, num, den = [[] for _ in range(Ka)], [0.0] * Ka, [0.0] * Ka
        AB = globals().get("FEAT_AUDIO_BATCH", 32)           # DAC at 44.1 kHz is heavy: keep the batch small on a small GPU
        for i in range(0, len(wav), AB):
            w = wav[i:i + AB].float() / 32768.0
            cl = codecs.audio.encode(w)
            for k in range(Ka):
                acc[k].append(cl[k])
            if hasattr(codecs.audio, "decode_mel") or getattr(codecs.audio, "decodable", False):
                ref = helper.logmel(w)
                for k in range(Ka):
                    z = codecs.audio.decode_mel(cl, level=k) if hasattr(codecs.audio, "decode_mel") else helper.logmel(codecs.audio.decode(cl, level=k).float())
                    num[k] += float(((z - ref) ** 2).sum()); den[k] += float(((ref - ref.mean()) ** 2).sum())
        rec = [1 - num[k] / max(den[k], 1e-9) for k in range(Ka)]
        print(f"AUDIO: {len(wav)} clips | log-mel R^2 of the decoded audio from the codes of levels 0..k (1.0 = perfect): " + _fmt_levels(rec, "{:.2f}"))
        if not all(math.isfinite(r) for r in rec):
            _flag("HARD", "audio reconstruction metric is not finite")
        elif rec[-1] < 0.5:
            _flag("SOFT", f"audio finest-level log-mel R^2 {rec[-1]:.2f} is low")
        CODEC_REPORT["audio"] = dict(kind="r2", recon=rec, rows=_analyse("AUDIO", "audio", [torch.cat(a) for a in acc]))
        del wav
    except Exception as e:
        _flag("HARD", f"audio codec failed: {type(e).__name__}: {str(e)[:200]}")
    # ---------------- text ----------------
    try:
        rows = collect_rows(("text0",), {"text0": CHK_TEXT}, 1, PARTS, WORKERS)
        b = torch.from_numpy(np.stack([r["tgt"] for r in rows])).to(dev); del rows
        cl = codecs.text.encode(b)
        exact = TextCodec.decode(cl[-1][:64]) == TextCodec.decode(b[:64].long() + 1)     # decode() subtracts 1: codes are byte + 1
        print(f"\nTEXT: {len(b)} rows | finest level decodes to the exact bytes: {exact}")
        if not exact:
            _flag("HARD", "text: the finest level does not decode to the input bytes")
        CODEC_REPORT["text"] = dict(exact=exact, rows=_analyse("TEXT", "text", cl))
    except Exception as e:
        _flag("HARD", f"text codec failed: {type(e).__name__}: {str(e)[:200]}")
    # ---------------- gate ----------------
    hard = [m for s, m in CODEC_FLAGS if s == "HARD"]
    soft = [m for s, m in CODEC_FLAGS if s == "SOFT"]
    gc.collect()
    if IS_CUDA:
        torch.cuda.empty_cache()                             # the probes and the check batches must not sit in the cache next to the model
    print(f"\nCODEC CHECKS: {len(hard)} hard failure(s), {len(soft)} warning(s)")
    if hard and CODEC_GATE == "stop":
        raise RuntimeError("codec checks failed: " + "; ".join(hard) + ". Fix the codec, or set CODEC_GATE='warn' to continue anyway.")


if CODEC_CHECK and "codecs" in globals():
    run_codec_checks()
