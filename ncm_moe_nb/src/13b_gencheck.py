# %% [markdown]
# ## 15b. Does it generate anything? (sanity checks, not a benchmark)
#
# **Text.** A character-bigram model fitted on real text scores four byte sets in bits per byte (lower = more like real text): real held-out
# text, the same bytes shuffled, bytes sampled from the training unigram, and model samples. Model samples below the shuffled / unigram
# controls have learned local structure; samples near the real text have learned a lot. A few samples are printed.
#
# **Image.** The DINO codec has no pixel decoder, so generated images cannot be shown. Instead each generated code map (level 2, a 4x4 layout)
# is turned back into DINO prototype features and compared with a gallery of real images: the mean best cosine similarity for (a) real
# images against the other real images, (b) model samples, (c) random codes drawn from the training unigram. Model samples near (a) and
# well above (c) are plausible image layouts; near (c) they are noise. The nearest real images of the first samples are displayed.
#
# Both use unconditional / generic prompts: caption and prompt following are checked separately in section 14b.

# %%
import math
from PIL import Image as PILImage


def _bigram(rows):
    a, b = rows[:, :-1].reshape(-1).long(), rows[:, 1:].reshape(-1).long()
    C = torch.bincount(a * 256 + b, minlength=65536).float().view(256, 256) + 0.1
    return (C / C.sum(1, keepdim=True)).log2()


def _bits(M, x):
    return float(-M[x[:, :-1].long(), x[:, 1:].long()].mean())


if not META_ONLY:
    # ---- text
    _rows = collect_rows(("text0",), {"text0": 2000}, 1, PARTS, WORKERS)
    M = _bigram(torch.from_numpy(np.stack([r["tgt"] for r in _rows])).to(dev))
    del _rows
    real = (torch.cat([mb["codes"][-1] for mb in VAL if mb["rtype"] == "text0"]) - 1).clamp(min=0)
    u = UNI[("text", 4)].float()
    uni = (torch.multinomial(u / u.sum(), real.numel(), replacement=True) - 1).clamp(min=0).view(real.shape)
    shuf = torch.gather(real, 1, torch.rand(real.shape, device=dev).argsort(1))
    gen = (generate("text0", None, n=64, temperature=0.9, top_k=64)[-1] - 1).clamp(min=0)
    print("TEXT: bigram bits per byte (lower = more like real text) and printable-ASCII share")
    for name, x in (("real held-out", real), ("shuffled real", shuf), ("unigram samples", uni), ("model samples", gen)):
        print(f"  {name:16s} {_bits(M, x):.2f} bits | printable {((x >= 32) & (x < 127)).float().mean():.2f}")
    for s in TextCodec.decode(gen[:3] + 1):
        print("  sample:", repr(s[:160]))

    # ---- does the prompt change the output? A prompt the model follows must move the bytes.
    # Two unrelated prompts yielding the same text is the cheapest possible proof that the
    # conditioning is inert - stronger than a CE gap, because it is what a user would see.
    try:
        _P = ["what is the capital of france?", "1 + 1 =",
              "the quick brown fox jumps over the lazy dog"]
        _outs = [TextCodec.decode((generate("text", p, n=1, temperature=0.9, top_k=64)[-1] - 1)
                                  .clamp(min=0) + 1)[0][:200] for p in _P]

        def _grams(s, k=4):
            return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}

        _sim = [len(_grams(_outs[i]) & _grams(_outs[j])) /
                max(1, len(_grams(_outs[i]) | _grams(_outs[j])))
                for i in range(len(_outs)) for j in range(i + 1, len(_outs))]
        print(f"TEXT: prompt sensitivity - 4-gram Jaccard between the outputs of {len(_P)} different "
              f"prompts: mean {float(np.mean(_sim)):.2f}  (0.0 = the prompt changes everything, "
              f"~1.0 = the prompt does nothing)")
        if not any(mb["rtype"] == "text" for mb in VAL):
            print("  !! this run had NO prompt->response rows (KEEP_INSTRUCT=0 and a prompt-free "
                  "text source), so the prompt never had a gradient: a high number here is the "
                  "configuration talking, not a model defect. Set NCM_KEEP_INSTRUCT=1 to test it.")
        for p, o in zip(_P, _outs):
            print(f"  {p!r:44s} -> {o[:90]!r}")
    except Exception as _e:                                   # noqa: BLE001
        print(f"  prompt sensitivity skipped: {type(_e).__name__}: {_e}")

    # ---- image (feature space: the DINO codec has no pixel decoder)
    if not getattr(codecs.image, "decodable", True):
        rows = collect_rows(("image",), {"image": 600}, 1, PARTS, WORKERS)
        gal = _img_tensor(rows)
        del rows

        def _norm(f):
            return F.normalize(F.normalize(f, dim=-1).flatten(1), dim=-1)

        with torch.no_grad():
            gf = torch.cat([_norm(codecs.image.features(gal[i:i + 32])[2]) for i in range(0, len(gal), 32)])
            q = _norm(codecs.image.reconstruct(codecs.image.encode(gal[:100])[2], 2))
            s = q @ gf.T
            s[torch.arange(100), torch.arange(100)] = -1
            real_top1 = float(s.max(1).values.mean())
            gcodes = generate("image", "a photo", n=32, temperature=0.9, top_k=64)[2]
            n2 = codecs.image.protos[2].shape[0]                            # level 2 has only n2 distinct codes, but the model can sample any of 256
            bad = float((gcodes > n2).float().mean())
            gcodes = gcodes.clamp(1, n2)
            qg = _norm(codecs.image.reconstruct(gcodes, 2))
            gen_top1 = float((qg @ gf.T).max(1).values.mean())
            ui = UNI[("image", 2)].float()
            rc = torch.multinomial(ui / ui.sum(), 32 * 16, replacement=True).view(32, 16)
            rand_top1 = float((_norm(codecs.image.reconstruct(rc, 2)) @ gf.T).max(1).values.mean())
        print(f"IMAGE: mean best cosine to a real image (level-2 DINO layout): real vs other real {real_top1:.3f} | model samples {gen_top1:.3f} | random codes {rand_top1:.3f}")
        print(f"  {100 * bad:.1f}% of the sampled level-2 codes were codes that never occur at that level ({n2} are valid); they were clamped")
        top = (qg[:6] @ gf.T).topk(4, dim=1).indices
        canvas = PILImage.new("RGB", (96 * 4, 96 * 6))
        for i in range(6):
            for j in range(4):
                canvas.paste(PILImage.fromarray(gal[top[i, j]].permute(1, 2, 0).cpu().numpy()).resize((96, 96)), (j * 96, i * 96))
        display(canvas)
        print("rows = the 4 nearest real images of the first 6 model samples")
