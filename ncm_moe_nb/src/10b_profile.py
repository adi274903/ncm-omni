# %% [markdown]
# ## 12b. Profile and A/B the speed levers (optional: `PROFILE=1`)
#
# Where does a training step spend its GPU time, and do the optional paths (`ATTN="levels"`, `MOE_DISPATCH="flat"`) help on THIS GPU?
# It profiles forward+backward per row type (kernel time grouped into attention / matmul / swiglu / dispatch+copies / optimizer / other),
# then times the attention variants and the two dispatch variants on the real micro-batch shapes and prints the knobs to set.
# Nothing here changes the model; run it once on a new GPU before a long run.

# %%
if not META_ONLY and PROFILE and IS_CUDA:
    import contextlib
    from torch.profiler import profile, ProfilerActivity

    def _cat(name):
        n = name.lower()
        if any(s in n for s in ("flash", "fmha", "attention", "sdpa", "flex", "efficient", "mha")):
            return "attention"
        if any(s in n for s in ("swiglu", "silu")):
            return "swiglu"
        if any(s in n for s in ("gemm", "cutlass", "cublas", "xmma", "nvjet", "ampere_", "hopper_", "sm80_", "sm90_", "sm100", "bmm")):
            return "matmul"
        if any(s in n for s in ("index", "gather", "scatter", "topk", "sort", "cat", "copy", "permute", "transpose", "memcpy")):
            return "dispatch / copies"
        if any(s in n for s in ("adam", "foreach", "multi_tensor")):
            return "optimizer"
        return "other (norms, elementwise, CE, ...)"

    def _kernel_us(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0

    def _profile_row(rtype, steps=3):
        B = MICRO_B[rtype]
        cond, codes = synth(rtype, B)
        net.train()
        for _ in range(2):                                           # warm-up (compile, autotune)
            with _amp(): loss, _ = loss_fn(rtype, cond, codes)          # bf16 autocast, exactly as the training step runs it
            loss.backward()
        net.zero_grad(set_to_none=True); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                with _amp(): loss, _ = loss_fn(rtype, cond, codes)
                loss.backward()
            torch.cuda.synchronize()
        net.zero_grad(set_to_none=True)
        by, top = {}, []
        for e in prof.key_averages():
            t = _kernel_us(e)
            if t:
                by[_cat(e.key)] = by.get(_cat(e.key), 0) + t
                top.append((t, e.key))
        tot = sum(by.values()) or 1
        print(f"  {rtype:6s} GPU kernel time per micro-batch {tot / steps / 1e3:7.1f} ms | " +
              " | ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
        return sorted(top, reverse=True)[:60]

    print("=== where the time goes (forward + backward, one micro-batch per row type) ===")
    tops = {r: _profile_row(r) for r in ROW_TYPES}
    for t, k in tops["image"][:6]:
        print(f"    image top kernel: {t / 3e3:7.2f} ms  {k[:90]}")
    for r_ in ("image", "audio"):                                     # what is hiding inside the copy bucket?
        print(f"    {r_}: biggest kernels in the dispatch / copies bucket:")
        for t, k in [x for x in tops[r_] if _cat(x[1]) == "dispatch / copies"][:8]:
            print(f"      {t / 3e3:7.2f} ms  {k[:110]}")

    def _time(fn, n=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1e3

    _a0, _m0 = ATTN, MOE_DISPATCH
    print("\n=== attention variants, one layer fwd+bwd (ms) ===")
    best_attn = {}
    for r in ROW_TYPES:
        lay, B = net.lay[r], MICRO_B[r]
        x0 = torch.randn(B, lay.T, P["d"], device=dev, dtype=AMP_DTYPE)
        res = {}
        from torch.nn.attention import sdpa_kernel, SDPBackend
        for name, use_flex in (("sdpa", False), ("flex", True), ("levels", False), ("levels+cudnn", False), ("levels+flash", False)):
            if name == "flex" and not getattr(net, "use_flex", False):
                continue
            ATTN = "levels" if name.startswith("levels") else "sdpa"
            _be = {"levels+cudnn": SDPBackend.CUDNN_ATTENTION, "levels+flash": SDPBackend.FLASH_ATTENTION}.get(name)
            def f(use_flex=use_flex, _be=_be):
                xx = x0.detach().requires_grad_(True)
                with _amp(), (sdpa_kernel(_be) if _be else contextlib.nullcontext()):
                    y = net.blocks[0].attn(xx, lay, use_flex)
                y.float().sum().backward()
            try:
                res[name] = _time(f)
            except Exception as e:
                print(f"    {r} {name}: failed ({type(e).__name__})")
        ATTN = _a0
        best_attn[r] = min(res, key=res.get)
        print(f"  {r:6s} " + " | ".join(f"{k} {v:.2f}" for k, v in res.items()) + f"   -> best {best_attn[r]}")

    print("\n=== expert dispatch, one MoE layer fwd+bwd (ms) ===")
    flat_wins = 0
    for r in ROW_TYPES:
        lay, B, m = net.lay[r], MICRO_B[r], ROW_MOD[r]
        x0 = torch.randn(B, lay.T, P["d"], device=dev, dtype=AMP_DTYPE)
        res = {}
        for name in ("gather", "flat"):
            MOE_DISPATCH = name
            def f():
                xx = x0.detach().requires_grad_(True)
                with _amp():
                    y = net.blocks[0].experts[m](xx, lay.segs[m])
                y.float().sum().backward()
            res[name] = _time(f)
        MOE_DISPATCH = _m0
        flat_wins += res["flat"] < res["gather"]
        print(f"  {r:6s} gather {res['gather']:.2f} | flat {res['flat']:.2f}")
    net.zero_grad(set_to_none=True)
    ATTN, MOE_DISPATCH = _a0, _m0
    pick_attn = max(set(best_attn.values()), key=list(best_attn.values()).count)
    print(f"\nper-layer timing favours: ATTN={pick_attn!r}, MOE_DISPATCH={'flat' if flat_wins >= 3 else 'gather'!r}  "
          f"(current: ATTN={ATTN!r}, MOE_DISPATCH={MOE_DISPATCH!r}). Differences under ~5% are noise.\n"
          f"  NOTE: this times ONE layer. On an RTX PRO 6000 it picked ATTN='sdpa' but a real run was 13% slower with it "
          f"(62k vs 72k codes/s): confirm any attention change with a short training run before adopting it.")
