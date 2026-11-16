# %% [markdown]
# ## 15. Generate: K forwards per sample, coarse to fine
#
# Level 0 is sampled first (from NULL input, given only the prefix), then each next level is sampled in
# one forward from the previous level's sampled codes. The prefix makes the same weights do
# **text -> text** (prompt), **text -> image** (caption) and **text -> speech** (transcript = basic TTS).
# The vocoder is Griffin-Lim on the decoded log-mel (positive NNLS spectrum inversion): intelligible at best, robotic by construction.
# A short run will produce babble; what these cells prove is that the whole conditional path works.
#
# **Guidance (`cfg`).** With caption dropout in training the model also knows the empty caption. At sampling time each level's
# log-probabilities are `uncond + cfg * (cond - uncond)`: `cfg=1` is plain sampling, larger values push the sample toward the caption.
# It costs two forwards per level. Decoding a level needs the codes of ALL levels up to it (the learned tokenizers are residual).

# %%
@torch.no_grad()
def generate(rtype, prompt=None, n=1, temperature=0.9, top_k=64, seed=0, cfg=1.0, diag=None):
    """`diag`: optional list; each level appends {k, distinct, p} so a caller can tell a collapsed
    hierarchy (1 code, p~1.0) from a healthy one (many codes, middling p) without its own loop."""
    net.eval()
    lay = LAYOUTS[rtype]
    g = torch.Generator(device=dev).manual_seed(seed)
    cond, blank = None, None
    if lay.cond:
        arr, _ = _bytes(prompt or "", SPEC.cond)
        cond = torch.from_numpy(arr)[None].repeat(n, 1).to(dev).long() + 1
        blank = torch.ones_like(cond)                                 # empty caption = all padding, exactly what caption dropout trains
    guided = cfg != 1.0 and cond is not None
    codes = [torch.zeros(n, l, dtype=torch.long, device=dev) for l in lay.lens]
    for k in range(lay.K):
        with _amp():
            h = net.hidden(rtype, cond, codes)
            hu = net.hidden(rtype, blank, codes) if guided else None
        lg = net.level_logits(rtype, h, k)
        if guided:
            lc, lu = lg.log_softmax(-1), net.level_logits(rtype, hu, k).log_softmax(-1)
            lg = lu + cfg * (lc - lu)
        lg = lg / temperature
        lg[..., 0] = float("-inf")                                    # NULL is never a sample
        if top_k:
            kth = lg.topk(top_k, dim=-1).values[..., -1:]
            lg = lg.masked_fill(lg < kth, float("-inf"))
        codes[k] = torch.multinomial(lg.softmax(-1).reshape(-1, SPEC.V), 1, generator=g).view(n, -1)
        if diag is not None:
            _p = lg.softmax(-1)
            diag.append(dict(k=k, distinct=int(codes[k].unique().numel()),
                             p=float(_p.max(-1).values.mean())))
    net.train()
    return codes


try:
    from IPython.display import display, Audio
except Exception:
    display = print; Audio = None


def show_text(prompt, **kw):
    kw.pop("cfg", None)                                               # guidance is for image / speech
    out = TextCodec.decode(generate("text", prompt, **kw)[-1])[0]
    print(f"[text->text]  {prompt!r}\n  -> {out!r}")


def show_image(caption, n=4, **kw):
    from PIL import Image as PILImage
    if not getattr(codecs.image, "decodable", True):
        print(f"[text->image] {caption!r}: skipped, the {type(codecs.image).__name__} codes have no pixel decoder "
              "(judge image learning from the held-out CE in the 'did it learn' cells)")
        return
    codes = generate("image", caption, n=n, **kw)
    rows = []
    for k in range(LAYOUTS["image"].K):                               # the coarse-to-fine progression, one row per level
        im = codecs.image.decode(codes, level=k).cpu().numpy()
        rows.append(np.concatenate([x.transpose(1, 2, 0) for x in im], 1))
    display(PILImage.fromarray(np.concatenate(rows, 0)))
    print(f"[text->image] {caption!r}: rows = level 0..{LAYOUTS['image'].K - 1} (blurry -> detailed), {n} samples")


def tts(text, **kw):
    codes = generate("audio", text, n=1, **kw)
    wav = codecs.audio.decode(codes, gl_iters=60)[0].float().cpu().numpy()
    wav = wav / max(1e-6, abs(wav).max()) * 0.9
    print(f"[text->speech] {text!r}: {len(wav)/SPEC.sr:.1f} s, " + ("Griffin-Lim vocoder" if hasattr(codecs.audio, "decode_mel") else "DAC decoder"))
    if Audio is not None:
        display(Audio(wav, rate=SPEC.sr))
    return wav


if not META_ONLY:
    gen_kw = dict(temperature=0.9, top_k=64)
    gen_g = dict(gen_kw, cfg=CFG_SCALE if CAP_DROP > 0 else 1.0)       # guided sampling for image / speech

    print("TEXT: one prompt at three temperatures. `Lk(n_codes,p)` = distinct codes sampled at that level")
    print("      and the model's mean top-1 probability there. A collapsed hierarchy shows 1 code at p~1.00;")
    print("      a healthy one samples many codes at a middling p. Low temperature exposes the dominant mode.")
    for _T in (0.2, 0.7, 1.0):
        _d = []
        _s = TextCodec.decode(generate("text", "what is the capital of france?",
                                       temperature=_T, top_k=64, diag=_d)[-1])[0]
        print("  T=%-4s %s" % (_T, " ".join("L%d(%dc,p%.2f)" % (x["k"], x["distinct"], x["p"]) for x in _d)))
        print("         %r" % _s[:130])

    print("\nTEXT: three prompts at T=0.7. Near-identical output means the prompt is doing nothing.")
    for _p in ("what is the capital of france?", "1 + 1 =", "the cat sat on the"):
        print("  %-34r -> %r" % (_p, TextCodec.decode(
            generate("text", _p, temperature=0.7, top_k=64)[-1])[0][:110]))

    print("\nIMAGE:")
    show_image("a photo of a dog on the grass", **gen_g)
    if not getattr(codecs.image, "decodable", True):
        print("       To actually LOOK at a generated image you need a pixel decoder: NCM_IMAGE_CODEC=vq")
        print("       (the learned multi-scale residual VQ-VAE, ~21.7 dB). Its codes differ from DINO's, so")
        print("       that needs a fresh codec fit AND a fresh training run - a DINO-trained checkpoint")
        print("       cannot be rendered by any decoder, and this is not a bug in the generation path.")

    print("\nSPEECH:")
    for _t in ("the quick brown fox jumps over the lazy dog", "hello world", "one two three four five"):
        _ = tts(_t, **gen_g)
