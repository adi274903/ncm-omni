# %% [markdown]
# ## 12. Correctness: fails loudly and cheaply, before any streamed token is spent
#
# 1. **No level leakage.** Perturbing the codes of level *j* leaves the hidden state of every slot at
#    levels <= *j* unchanged and moves level *j+1* (so the model cannot copy its own targets).
# 2. **The prefix is a real condition.** Perturbing the prompt/caption/transcript moves level 0, and
#    perturbing targets never moves the prefix.
# 3. **Modality isolation.** A text-only batch sends exactly zero gradient into the image and audio experts.
# 4. **Expert capacity.** Coverage stats: every token gets 0, 1 or 2 experts, about 1 on average.
# 5. **Kernel parity.** Triton SwiGLU vs reference, chunked CE vs `F.cross_entropy`, flex vs SDPA (printed above).
# 6. **Codec round trip** and an initial loss at the uniform-prediction value `ln(257)`.

# %%
def _fresh_codes(rtype, B, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    lay = LAYOUTS[rtype]
    codes = [torch.randint(1, 257, (B, n), device=dev, generator=g) for n in lay.lens]
    cond = torch.randint(1, 257, (B, lay.cond), device=dev, generator=g) if lay.cond else None
    return cond, codes


@torch.no_grad()
def check_no_leak(net, rtype):
    lay = LAYOUTS[rtype]
    cond, codes = _fresh_codes(rtype, 2, 1)
    h0 = net.hidden(rtype, cond, codes)
    worst = {}
    for j in range(lay.K):
        codes2 = [c.clone() for c in codes]
        codes2[j] = (codes2[j] % 256) + 1                                  # every code at level j changes
        h1 = net.hidden(rtype, cond, codes2)
        upto = lay.lvl[j][1]
        same = (h0[:, :upto] - h1[:, :upto]).abs().max().item()            # prefix + levels <= j
        assert same < 1e-5, f"{rtype}: perturbing level {j} moved a slot at level <= {j} by {same:.2e} (LEAK)"
        if j + 1 < lay.K:
            a, b = lay.lvl[j + 1]
            moved = (h0[:, a:b] - h1[:, a:b]).abs().max().item()
            assert moved > 1e-4, f"{rtype}: level {j+1} ignores level {j} ({moved:.2e})"
            worst[j] = round(moved, 3)
    if lay.cond:
        cond2 = (cond % 256) + 1
        h1 = net.hidden(rtype, cond2, codes)
        a, b = lay.lvl[0]
        moved = (h0[:, a:b] - h1[:, a:b]).abs().max().item()
        assert moved > 1e-4, f"{rtype}: level 0 ignores the prefix"
    return worst


if not META_ONLY:
    net.eval()
    for r in ROW_TYPES:
        w = check_no_leak(net, r)
        print(f"  no-leak ok: {r:6s} level->level influence { w }")

    # (6) initial loss: heads start at zero, so the first prediction is uniform and the loss is ln(257)
    net.eval()
    with torch.no_grad():
        l0, _ = net.loss("text0", *_fresh_codes("text0", 2, 2))
    assert abs(l0.item() - math.log(257)) < 0.05, f"init loss {l0.item():.3f} != ln(257)={math.log(257):.3f}"
    print(f"  init loss {l0.item():.3f} ~ ln(257) = {math.log(257):.3f}")

    # (3) modality isolation. A zero head sends exactly zero gradient into the trunk on step 0, so the test
    # swaps in a small random head, checks the gradient, and puts the zero head back.
    saved = {k: v.clone() for k, v in net.heads.state_dict().items()}
    for hd in net.heads.values():
        nn.init.normal_(hd.weight, std=0.02)
    net.train(); net.zero_grad(set_to_none=True)
    cond, codes = _fresh_codes("text0", 2, 2)
    loss, _ = net.loss("text0", cond, codes)
    loss.backward()
    g = {m: sum(float(p.grad.abs().sum()) if p.grad is not None else 0.0
                for blk in net.blocks for p in blk.experts[m].parameters()) for m in ("text", "image", "audio")}
    assert g["text"] > 0 and g["image"] == 0 and g["audio"] == 0, f"modality isolation broken: {g}"
    print("  isolation ok: text0 batch ->", {m: ("grad" if v > 0 else "zero grad") for m, v in g.items()})
    net.zero_grad(set_to_none=True)
    net.heads.load_state_dict(saved)

    # (4) coverage
    net.eval()
    for blk in net.blocks:
        for e in blk.experts.values():
            e.record = True
    with torch.no_grad():
        net.hidden("image", *_fresh_codes("image", 4, 3)[:1], _fresh_codes("image", 4, 3)[1])
    cov = {m: net.blocks[-1].experts[m].last_cov for m in ("text", "image")}
    for blk in net.blocks:
        for e in blk.experts.values():
            e.record = False
    for m, c in cov.items():
        if c is not None:
            mean = float(c[1] + 2 * c[2])
            print(f"  coverage (last block, {m} group): 0 experts {c[0]:.2f} | 1 expert {c[1]:.2f} | 2 experts {c[2]:.2f} | mean {mean:.2f}")
            assert 0.9 < mean < 1.3

    # (5) kernels
    h = torch.randn(64, 2 * 48, device=dev, dtype=torch.float32, requires_grad=True)
    ref = swiglu_torch(h)
    assert torch.allclose(swiglu(h), ref, atol=1e-4, rtol=1e-4)
    W, b_ = torch.randn(257, 32, device=dev), torch.randn(257, device=dev)
    hh, tt = torch.randn(1000, 32, device=dev), torch.randint(1, 257, (1000,), device=dev)
    _tf32 = torch.backends.cuda.matmul.allow_tf32      # TF32 rounds the two matmuls differently (~1e-2), which hides real bugs
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ce = fused_ce(hh, W, b_, tt, chunk=300)[0]
        ref_ce = F.cross_entropy(F.linear(hh.double(), W.double(), b_.double()), tt, reduction="none")
        err = (ce.double() - ref_ce).abs().max().item()
        assert err < 1e-3, f"chunked CE differs from reference by {err:.2e}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32
    print(f"  kernels ok: swiglu[{SWIGLU_IMPL}] == reference, chunked CE == F.cross_entropy")

    # (6) codecs
    txt = torch.randint(0, 256, (2, SPEC.text), dtype=torch.uint8, device=dev)
    assert TextCodec.decode(codecs.text.encode(txt)[-1])[0] == bytes(txt[0].tolist()).rstrip(b"\x00").decode("utf-8", "replace")
    print("  codec round trip ok (text exact); sizes:", {r: LAYOUTS[r].Tt for r in ROW_TYPES})
    # image / audio tokenizers: right shapes, codes in 1..256, decode gives the right size at every level (works for k-means and learned)
    _img = torch.randint(0, 256, (2, 3, SPEC.side, SPEC.side), dtype=torch.uint8, device=dev)
    _ci = codecs.image.encode(_img)
    assert [c.shape[1] for c in _ci] == LAYOUTS["image"].lens and all(1 <= int(c.min()) and int(c.max()) <= 256 for c in _ci), "image codes: wrong shape or range"
    if getattr(codecs.image, "decodable", True):                      # the DINO feature codec has no pixel decoder
        assert all(codecs.image.decode(_ci, level=k).shape == _img.shape for k in range(LAYOUTS["image"].K)), "image decode: wrong size"
    _wv = torch.randn(2, SPEC.wave_len, device=dev) * 0.1
    _ca = codecs.audio.encode(_wv)
    assert [c.shape[1] for c in _ca] == LAYOUTS["audio"].lens and all(1 <= int(c.min()) and int(c.max()) <= 256 for c in _ca), "audio codes: wrong shape or range"
    if hasattr(codecs.audio, "decode_mel"):
        assert all(codecs.audio.decode_mel(_ca, level=k).shape == (2, SPEC.audio, SPEC.n_mels) for k in range(LAYOUTS["audio"].K)), "audio decode_mel: wrong shape"
    else:                                                            # the DAC codec decodes straight to a waveform
        assert all(codecs.audio.decode(_ca, level=k).shape == (2, SPEC.wave_len) for k in range(LAYOUTS["audio"].K)), "audio decode: wrong size"
    print(f"  tokenizer round trip ok ({TOKENIZER}): image {[tuple(c.shape[1:]) for c in _ci]}, audio {[tuple(c.shape[1:]) for c in _ca]}")

    # (7) the optional speed paths must equal the reference: level-wise attention vs masked SDPA, flat expert dispatch vs gather/scatter
    _a0, _m0 = ATTN, MOE_DISPATCH
    try:
        net.eval()
        for r in ROW_TYPES:
            lay = net.lay[r]
            x = torch.randn(2, lay.T, P["d"], device=dev, dtype=AMP_DTYPE)
            with torch.no_grad(), _amp():
                ATTN = "sdpa";   a_ref = net.blocks[0].attn(x, lay, False).float()
                ATTN = "levels"; a_lv = net.blocks[0].attn(x, lay, False).float()
                ATTN = _a0
                MOE_DISPATCH = "gather"; m_ref = sum(net.blocks[0].experts[g](x, s) for g, s in lay.segs.items()).float()
                MOE_DISPATCH = "flat";   m_fl = sum(net.blocks[0].experts[g](x, s) for g, s in lay.segs.items()).float()
            ea, em = (a_ref - a_lv).abs().max().item(), (m_ref - m_fl).abs().max().item()
            assert ea < 5e-2, f"{r}: level-wise attention differs from masked SDPA by {ea:.2e}"
            assert em < 5e-2, f"{r}: flat expert dispatch differs from gather/scatter by {em:.2e}"
        print(f"  speed paths ok: ATTN='levels' == masked SDPA, MOE_DISPATCH='flat' == gather/scatter (last max abs err {ea:.1e} / {em:.1e}); active: ATTN={_a0}, MOE_DISPATCH={_m0}")
    finally:
        ATTN, MOE_DISPATCH = _a0, _m0
        net.train()
    print("\nALL CORRECTNESS CHECKS PASSED")
    net.train()
