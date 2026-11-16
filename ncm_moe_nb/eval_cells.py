# Evaluation cells for the NCM MoE notebook (learned-tokenizer version). Paste CELL 1, CELL 2, CELL 3 as new code cells AFTER the
# generate cell, same kernel: they use net, codecs, VAL, SPEC, LAYOUTS, ROW_MOD, UNI, generate, evaluate, _amp, dev, CFG_SCALE, CAP_DROP
# from the notebook. Nothing here trains or writes to Drive. CELL 1 needs no extra packages; CELL 2 downloads Whisper-small and CLIP-B/32.
# Codes are RESIDUAL with the learned tokenizers: decode(codes_list, level=k) needs all levels <= k, so every decode below gets the list.


# %% ============================ CELL 1: likelihood, tokenizer ceilings, reconstruction, caption gap, samples ============================
import math, numpy as np, torch, torch.nn.functional as F
LN2, N_SAMP = math.log(2), 24
G_KW = dict(temperature=0.9, top_k=64)
G_CFG = dict(G_KW, cfg=CFG_SCALE if CAP_DROP > 0 else 1.0)         # guided sampling (only meaningful when trained with caption dropout)


def _ci(x, n=1000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _txt(c):
    return bytes((c - 1).clamp(0, 255).to(torch.uint8).cpu().tolist()).rstrip(b"\x00").decode("utf-8", "replace")


def _psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1).clamp(min=1e-8)
    return 10 * torch.log10(255.0 ** 2 / mse)


@torch.no_grad()
def forward_stats(r, cond, codes):
    """Teacher-forced pass. Returns per-row summed CE in nats over ALL levels at live (non-NULL) targets, and finest-level logits."""
    with _amp():
        h = net.hidden(r, cond, codes)
    tot, lg = 0.0, None
    for k in range(LAYOUTS[r].K):
        lg = net.level_logits(r, h, k).float()
        tg = codes[k]
        ce = F.cross_entropy(lg.reshape(-1, SPEC.V), tg.reshape(-1), reduction="none").view(tg.shape)
        tot = tot + (ce * (tg > 0).float()).sum(1)
    return tot, lg


def _mode_code(m):
    """Most frequent finest-level code: the 'predict the same thing everywhere' baseline for reconstruction."""
    h = UNI.get((m, SPEC.K - 1))
    if h is None:                                                   # UNI is not checkpointed: fall back to the held-out histogram
        h = sum(torch.bincount(mb["codes"][-1].reshape(-1), minlength=SPEC.V).float() for mb in VAL if ROW_MOD[mb["rtype"]] == m)
    h = h.clone(); h[0] = 0
    return int(h.argmax())


net.eval()
MODE = {m: _mode_code(m) for m in ("image", "audio")}
R = {}                                                              # metric name -> per-row values
def add(k, v): R.setdefault(k, []).extend(float(x) for x in v)

for mb in VAL:
    r, m = mb["rtype"], ROW_MOD[mb["rtype"]]
    cond, codes = mb["cond"], mb["codes"]
    tot, lg = forward_stats(r, cond, codes)
    live_fin = (codes[-1] > 0).sum(1).float()
    den = (live_fin if m == "text" else sum((c > 0).sum(1) for c in codes).float()) * LN2     # text: per byte; others: per code
    add(f"{r} | {'bits/byte' if m == 'text' else 'bits/code'}", tot / den)
    if cond is not None and cond.shape[0] > 1:                                                # does the prefix matter?
        t_b, _ = forward_stats(r, torch.ones_like(cond), codes)                              # blank = all padding (code 1), what caption dropout trains
        t_s, _ = forward_stats(r, cond.roll(1, 0), codes)                                    # shuffled = another row's caption (the clean test)
        add(f"{r} | gap blank-real (bits)", (t_b - tot) / den)
        add(f"{r} | gap shuffled-real (bits)", (t_s - tot) / den)
    lg = lg.clone(); lg[..., 0] = float("-inf")
    pred = lg.argmax(-1)
    if m == "image":                                                # teacher-forced argmax at the finest level, coarse levels true
        ref = codecs.image.decode(codes)
        add("image | PSNR model vs tokenizer-recon (dB)", _psnr(codecs.image.decode(codes[:-1] + [pred]), ref))
        add("image | PSNR same-code baseline (dB)", _psnr(codecs.image.decode(codes[:-1] + [torch.full_like(pred, MODE["image"])]), ref))
    if m == "audio":
        live = (codes[-1] > 0).float()
        zt = codecs.audio.decode_mel(codes)
        def rm(p): return (((codecs.audio.decode_mel(codes[:-1] + [p]) - zt) ** 2).mean(-1) * live).sum(1).div(live.sum(1).clamp(min=1)).sqrt()
        add("audio | log-mel RMSE model vs tokenizer-recon", rm(pred))
        add("audio | log-mel RMSE same-code baseline", rm(torch.full_like(pred, MODE["audio"])))

print("=== held-out, teacher-forced (mean, 95% bootstrap CI over rows) ===")
for k in sorted(R):
    mu, lo, hi = _ci(R[k]); print(f"  {k:48s} {mu:8.3f}  [{lo:.3f}, {hi:.3f}]  n={len(R[k])}")
for rt in ("text0", "text"):
    k = f"{rt} | bits/byte"
    if k in R:
        print(f"  {rt}: perplexity per byte <= {2 ** np.mean(R[k]):.3f}  (upper bound: coarse levels are functions of the bytes, so the sum over levels is a valid code length)")
print("\nper-level CE / constant baseline / top-1 (from evaluate()):")
for key, d in sorted(evaluate().items()):
    print(f"  {key:10s} ce {d['ce']:.3f}  base {d['base']:.3f}  gain {d['base'] - d['ce']:+.3f}  acc {100 * d['acc']:.1f}%")

# ---- tokenizer ceiling on RAW held-out data (part 0): what ANY model could reach with these codes ----
print("\n=== tokenizer ceiling vs raw (real data -> codes -> decode) ===")
try:
    rr = collect_rows(("image",), {"image": 64}, 0, PARTS, 1, seed=777)
    tmp = [None] * len(rr); jp = [j for j, x in enumerate(rr) if "jpeg" in x]
    if jp:
        d = decode_jpeg_gpu([rr[j]["jpeg"] for j in jp], SPEC.side, dev)
        for n_, j in enumerate(jp): tmp[j] = d[n_]
    for j, x in enumerate(rr):
        if tmp[j] is None: tmp[j] = torch.from_numpy(x["img"]).to(dev)
    X = torch.stack(tmp); enc = codecs.image.encode(X)
    print("  image PSNR vs raw, decoded at level 0..K-1:", [round(float(_psnr(codecs.image.decode(enc, level=j), X).mean()), 2) for j in range(SPEC.K)], "dB   (old k-means codec: 14.01 14.61 15.71 17.18 19.97)")
except Exception as e:
    print("  image ceiling skipped:", repr(e)[:150])
try:
    rr = collect_rows(("audio",), {"audio": 48}, 0, PARTS, 1, seed=777)
    w = torch.from_numpy(np.stack([x["wav"] for x in rr])).to(dev).float() / 32768
    target, enc_a = codecs.audio.logmel(w), codecs.audio.encode(w)
    print("  audio log-mel RMSE vs raw, decoded at level 0..K-1:",
          [round(float(((codecs.audio.decode_mel(enc_a, level=j) - target) ** 2).mean().sqrt()), 3) for j in range(SPEC.K)], "  (old k-means codec: finest 0.984)")
except Exception as e:
    print("  audio ceiling skipped:", repr(e)[:150])


# ---- free-running samples (the real generation path, K forwards) ----
def val_prompts(rtype, n):
    out = []
    for mb in VAL:
        if mb["rtype"] == rtype and mb["cond"] is not None:
            for i in range(mb["cond"].shape[0]):
                out.append((_txt(mb["cond"][i]), [c[i:i + 1] for c in mb["codes"]], len(_txt(mb["cond"][i]).encode())))
                if len(out) >= n: return out
    return out

from PIL import Image as _PIL
IMAGE_SAMPLES, AUDIO_SAMPLES = [], []
for i, (cap, true_codes, _) in enumerate(val_prompts("image", N_SAMP)):
    g = generate("image", cap, n=1, seed=i, **G_CFG)
    IMAGE_SAMPLES.append((cap, codecs.image.decode(g)[0], codecs.image.decode(true_codes)[0]))
tiles = [np.concatenate([a.cpu().numpy().transpose(1, 2, 0), b.cpu().numpy().transpose(1, 2, 0)], 1) for _, a, b in IMAGE_SAMPLES[:8]]
display(_PIL.fromarray(np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles) - 1, 2)], 0)))
print(f"image samples (cfg {G_CFG['cfg']}): each pair = [generated | tokenizer reconstruction of the real image]")
for cap, _, _ in IMAGE_SAMPLES[:8]: print("  ", repr(cap[:90]))

for i, (cap, true_codes, ncond) in enumerate(val_prompts("audio", 8)):
    g = generate("audio", cap, n=1, seed=i, **G_CFG)
    gen = codecs.audio.decode(g, gl_iters=60)[0].float().cpu().numpy()
    ref = codecs.audio.decode(true_codes, gl_iters=60)[0].float().cpu().numpy()
    AUDIO_SAMPLES.append(dict(text=cap, gen=gen, ref=ref, full=ncond < SPEC.cond))      # full=False: transcript was cut to the prefix size
for s in AUDIO_SAMPLES[:3]:
    print(repr(s["text"][:90])); print("  generated:"); display(Audio(s["gen"] / max(1e-6, abs(s["gen"]).max()) * 0.9, rate=SPEC.sr))
    print("  real clip through the same tokenizer + Griffin-Lim:"); display(Audio(s["ref"] / max(1e-6, abs(s["ref"]).max()) * 0.9, rate=SPEC.sr))
for p_, _, _ in val_prompts("text", 3): show_text(p_, **G_KW)
net.train()


# %% ============================ CELL 2: WER (Whisper) for speech and CLIPScore for images. Downloads two small models ============================
import re
from transformers import pipeline, CLIPModel, CLIPProcessor


def _words(s): return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()
def _wer(ref, hyp):
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def _live_samples(w, frame=512):
    """Samples up to the last non-silent frame (+0.25 s), so Whisper is not asked to transcribe padding (it hallucinates on silence)."""
    e = np.sqrt(np.convolve(w.astype(np.float64) ** 2, np.ones(frame) / frame, mode="same"))
    idx = np.where(e > 0.02 * max(e.max(), 1e-8))[0]
    return min(len(w), int(idx[-1]) + SPEC.sr // 4) if len(idx) else len(w)


# same clips, same transcripts, three audio sources: the original, the tokenizer's own reconstruction (ceiling), the model's sample
rr = collect_rows(("audio",), {"audio": 24}, 0, PARTS, 1, seed=777)
rr = [x for x in rr if len(bytes(x["cond"]).rstrip(b"\x00")) < SPEC.cond][:20]                  # only transcripts that fit the prefix
texts = [bytes(x["cond"]).rstrip(b"\x00").decode("utf-8", "replace") for x in rr]
raw = torch.from_numpy(np.stack([x["wav"] for x in rr])).to(dev).float() / 32768
ceil = codecs.audio.decode(codecs.audio.encode(raw), gl_iters=60).float().cpu().numpy()
raw = raw.cpu().numpy()
net.eval()
gen = [codecs.audio.decode(generate("audio", t, n=1, seed=i, **G_CFG), gl_iters=60)[0].float().cpu().numpy() for i, t in enumerate(texts)]
net.train()

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if IS_CUDA else -1)
tr = lambda w: asr({"raw": w.astype(np.float32), "sampling_rate": SPEC.sr}, generate_kwargs={"language": "en"})["text"]
res = {"original clip": [], "tokenizer + Griffin-Lim (ceiling)": [], "model sample": []}
for i, t in enumerate(texts):
    L = _live_samples(raw[i])
    for name, w in (("original clip", raw[i]), ("tokenizer + Griffin-Lim (ceiling)", ceil[i]), ("model sample", gen[i])):
        h = tr(w[:min(L, len(w))]); res[name].append((_wer(t, h), len(_words(h)) / max(1, len(_words(t)))))
print(f"WER on {len(texts)} clips (trimmed to the live part; second number = hypothesis words / reference words, ~1 is healthy, >>1 = hallucinated words)")
for name, v in res.items():
    print(f"  {name:36s} WER {np.mean([a for a, _ in v]):.3f} | length ratio {np.mean([b for _, b in v]):.2f}")
print("  read: 'original clip' must be low (a few %), otherwise the measurement is broken. ceiling - original = what tokenizer + vocoder cost; sample - ceiling = what the model adds.")
del asr

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def clip_score(caps, imgs):
    x = proc(text=list(caps), images=[_PIL.fromarray(i.cpu().numpy().transpose(1, 2, 0)) for i in imgs], return_tensors="pt", padding=True, truncation=True).to(dev)
    o = clip(**x)
    ie, te = F.normalize(o.image_embeds, dim=-1), F.normalize(o.text_embeds, dim=-1)
    return 2.5 * (ie * te).sum(-1).clamp(min=0)                     # CLIPScore (Hessel et al.): 2.5 * max(cos, 0)


caps = [c for c, _, _ in IMAGE_SAMPLES]
gen_i, ref_i = [g for _, g, _ in IMAGE_SAMPLES], [r_ for _, _, r_ in IMAGE_SAMPLES]
sh = caps[1:] + caps[:1]
print(f"CLIPScore on {len(caps)} images | generated+its caption: {clip_score(caps, gen_i).mean():.3f} | generated+WRONG caption: {clip_score(sh, gen_i).mean():.3f} "
      f"| real image via tokenizer+its caption (ceiling): {clip_score(caps, ref_i).mean():.3f} | real+WRONG caption: {clip_score(sh, ref_i).mean():.3f}")
print("  read: only the matched-minus-wrong gap means anything; the absolute value depends on image size and CLIP.")
del clip


# %% ============================ CELL 3: guidance / sampling test for images, and where the robotic audio comes from (needs CELL 1 first) ============================
net.eval()

# ---- images: same captions; columns = [T0.9/k64 no guidance s0 s1 | T0.9/k64 guided s0 s1 | T0.5/k8 guided s0 s1] ----
CFG_ON = CFG_SCALE if CAP_DROP > 0 else 1.0
SETTINGS = [dict(temperature=0.9, top_k=64, cfg=1.0), dict(temperature=0.9, top_k=64, cfg=CFG_ON), dict(temperature=0.5, top_k=8, cfg=CFG_ON)]
rows_ = []
for cap, _, _ in IMAGE_SAMPLES[:4]:
    cells = [codecs.image.decode(generate("image", cap, n=1, seed=sd, **kw))[0].cpu().numpy().transpose(1, 2, 0) for kw in SETTINGS for sd in (0, 1)]
    rows_.append(np.concatenate(cells, 1))
display(_PIL.fromarray(np.concatenate(rows_, 0)))
print("rows = captions:"); [print("  ", repr(c[:90])) for c, _, _ in IMAGE_SAMPLES[:4]]
print(f"read: guided (cfg {CFG_ON}) should follow the caption more than unguided; similar subjects across seeds = real conditioning; speckle that vanishes at T0.5 is sampling noise.")

# ---- audio: raw clip -> log-mel -> (optional tokenizer) -> spectrum (old pinv vs new NNLS) -> Griffin-Lim ----
A = codecs.audio
rr = collect_rows(("audio",), {"audio": 6}, 0, PARTS, 1, seed=777)
w = torch.from_numpy(np.stack([x["wav"] for x in rr])).to(dev).float() / 32768
lm = A.logmel(w)                                                    # natural-log mel of the REAL clip [B, F, 80]
zq = A.decode_mel(A.encode(w))                                      # the same clip after the tokenizer (k-means or learned)


def vocode(logmel, iters, inv):
    global AUDIO_INV
    old, AUDIO_INV = AUDIO_INV, inv
    try:
        mag = A.mel_to_mag(torch.exp(logmel))
    finally:
        AUDIO_INV = old
    return griffin_lim(mag, A.win, SPEC.hop, A.N_FFT, iters)


def self_err(wav, target):
    """How far the vocoder output's OWN log-mel is from the mel it was asked to produce (natural-log units; 0 = perfectly consistent)."""
    a = A.logmel(wav); f = min(a.shape[1], target.shape[1])
    return float(((a[:, :f] - target[:, :f]) ** 2).mean().sqrt())


variants = [("original", None, None, None),
            ("raw mel, OLD pinv, GL 60", lm, 60, "pinv"), ("raw mel, NNLS, GL 60", lm, 60, "nnls"), ("raw mel, NNLS, GL 300", lm, 300, "nnls"),
            ("tokenizer codes, NNLS, GL 60", zq, 60, "nnls"), ("tokenizer codes, NNLS, GL 300", zq, 300, "nnls")]
for name, z, it, inv in variants:
    wav = w if z is None else vocode(z, it, inv)
    print(f"{name:32s}" + ("" if z is None else f" self-consistency error vs its own target mel: {self_err(wav, z):.3f}"))
    display(Audio(wav[0].float().cpu().numpy() / max(1e-6, float(wav[0].abs().max())) * 0.9, rate=SPEC.sr))
print("read: NNLS should have a lower self-consistency error and sound less metallic than the old pinv path. "
      "If 'raw mel, NNLS' is still metallic, the vocoder (Griffin-Lim) is the limit and a neural vocoder is the next step.")
net.train()
