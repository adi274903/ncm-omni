# %% [markdown]
# ## 10. Build and size check (training only; skipped when `SHARD_MODE="encode"`)
#
# The parameter report comes from a **meta-device** build (no memory), so it prints even on a machine that
# cannot hold the model. The real build then checks that fp32 weights + AdamW (16 bytes/param) fit.

# %%
net_meta = build_model(meta=True)
REPORT = param_report(net_meta)
del net_meta
M = 1e6
print(f"total {REPORT['total']/M:,.1f}M | trunk {REPORT['trunk']/M:.1f}M | experts {REPORT['experts_total']/M:,.1f}M "
      f"= 6 x {REPORT['per_expert']/M:.1f}M")
print("active per token:", {m: f"{v/M:.1f}M" for m, v in REPORT["active"].items()},
      f"| capacity {MOE_CF} x 2 experts = {2*MOE_CF:.1f} active expert(s)/token")
if PRESET == "full":
    assert abs(REPORT["total"] - 1.68e9) / 1.68e9 < 0.05, "full preset drifted from the 1.2B+ / ~360M-active target"
    assert 330e6 < REPORT["active_max"] < 390e6, f"active params {REPORT['active_max']/M:.0f}M"
need_gb = REPORT["total"] * 16 / 2 ** 30
print(f"fp32 weights + grads + AdamW ~ {need_gb:.1f} GiB before activations | device has {VRAM_GB:.0f} GiB")
META_ONLY = _knob("META_ONLY", False) or SHARD_MODE == "encode"
if IS_CUDA and not META_ONLY and VRAM_GB < need_gb * 1.1:
    raise SystemExit(f"PRESET='{PRESET}' needs ~{need_gb:.0f} GiB of VRAM; this GPU has {VRAM_GB:.0f}. "
                     f"Use PRESET='tiny' or an A100/H100 (see the budget cell).")

# %%
if not META_ONLY:
    net = build_model().to(dev)
    net.lay = {r: LAYOUTS[r].to(dev) for r in ROW_TYPES}
    # EVAL_ONLY must restore the WEIGHTS, not just the codebooks: `train()` is the only other place
    # that loads RESUME["net"], and EVAL_ONLY skips it - so without this an "evaluate the"
    # "checkpoint" run scores a freshly initialised model and prints numbers that look real.
    if EVAL_ONLY:
        if RESUME is not None:
            net.load_state_dict(RESUME["net"])
            print(f"EVAL_ONLY: weights restored from {CKPT_PATH} (step {RESUME['step']}, "
                  f"codes seen {sum(RESUME['seen'].values())/1e9:.4f}B)")
        else:
            print("!! EVAL_ONLY=1 but no checkpoint at CKPT_PATH: everything below runs on a RANDOM "
                  "model. Set NCM_CKPT_DIR to the folder that holds ckpt.pt.")
    torch._dynamo.config.cache_size_limit = 64

    def _try_flex():
        """Use flex-attention only if it compiles, matches SDPA, and is not slower."""
        from torch.nn.attention.flex_attention import create_block_mask
        for l in net.lay.values():
            bid = l.block_id
            l.flex_mask = create_block_mask(lambda b, h, q, kv, bid=bid: bid[kv] <= bid[q], None, None, l.T, l.T, device=dev)
        blk = net.blocks[0].attn
        r = "audio"
        x = torch.randn(2, net.lay[r].T, P["d"], device=dev, dtype=AMP_DTYPE)
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
            a = blk(x, net.lay[r], False).float()
            b = blk(x, net.lay[r], True).float()
        ok = torch.allclose(a, b, atol=5e-2, rtol=5e-2)
        for rr in ROW_TYPES:                                   # every packed geometry, forward AND backward: each compiles its own kernel
            xg = torch.randn(2, net.lay[rr].T, P["d"], device=dev, dtype=AMP_DTYPE, requires_grad=True)
            with torch.enable_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
                blk(xg, net.lay[rr], True).float().sum().backward()
        net.zero_grad(set_to_none=True)
        def bench(use):
            for _ in range(3):
                blk(x, net.lay[r], use)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10):
                blk(x, net.lay[r], use)
            torch.cuda.synchronize()
            return (time.time() - t) / 10
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
            ts, tf = bench(False), bench(True)
        print(f"attention: sdpa {ts*1e3:.2f} ms | flex {tf*1e3:.2f} ms | parity {ok}")
        return ok and (ATTN == "flex" or tf <= ts * 1.05)

    net.use_flex = False
    _smem_kb = getattr(torch.cuda.get_device_properties(0), "shared_memory_per_block_optin", 0) // 1024 if IS_CUDA else 0   # missing on ROCm
    if IS_CUDA and HAS_FLEX and ATTN == "auto" and _smem_kb < 160:
        print(f"flex_attention skipped: {_smem_kb} KB shared memory per block (A100-class has 163); using sdpa. ATTN=flex forces it.")
    elif IS_CUDA and HAS_FLEX and ATTN in ("auto", "flex"):
        try:
            net.use_flex = _try_flex()
        except Exception as e:
            print("flex_attention unavailable:", type(e).__name__, str(e)[:80])
    print("attention backend:", "levels (one unmasked flash call per level)" if ATTN == "levels" else ("flex (block mask)" if net.use_flex else "sdpa (bool mask)"),
          "| expert dispatch:", MOE_DISPATCH)

    COMPILED = bool(COMPILE and IS_CUDA)
    loss_fn = torch.compile(net.loss, dynamic=False, mode=COMPILE_MODE) if COMPILED else net.loss
    print(f"model on {dev} | {sum(p.numel() for p in net.parameters())/M:,.1f}M params | torch.compile {COMPILED}")
