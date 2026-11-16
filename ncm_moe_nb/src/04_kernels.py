# %% [markdown]
# ## 8. Kernels (training only; the shard path above never uses them)
#
# | op | implementation |
# |---|---|
# | SwiGLU `silu(a) * b` on the fused gate+up projection | Triton: one read, one write, fused backward (registered as a `torch.library` custom op so `torch.compile` can see through it) |
# | RMSNorm, residual adds, QK-norm | `F.rms_norm` + Inductor fusion |
# | attention | flex-attention block mask (level-block-causal) -> SDPA with a bool mask -> CPU math |
# | head matmul + cross-entropy | chunked and **checkpointed**: the `[tokens, V]` logits are never kept for backward |
# | expert GEMMs | both experts of a group run as one `bmm` (capacity is uniform, so a grouped GEMM *is* a batched GEMM) |
# | optimizer | `AdamW(fused=True)` |

# %%
def swiglu_torch(h):
    a, b = h.chunk(2, dim=-1)
    return F.silu(a) * b

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fwd_k(h_ptr, o_ptr, H, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        a = tl.load(h_ptr + row * 2 * H + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(h_ptr + row * 2 * H + H + cols, mask=m, other=0.0).to(tl.float32)
        o = a * tl.sigmoid(a) * b
        tl.store(o_ptr + row * H + cols, o.to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_bwd_k(h_ptr, g_ptr, d_ptr, H, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        a = tl.load(h_ptr + row * 2 * H + cols, mask=m, other=0.0).to(tl.float32)
        b = tl.load(h_ptr + row * 2 * H + H + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * H + cols, mask=m, other=0.0).to(tl.float32)
        s = tl.sigmoid(a)
        da = g * b * s * (1.0 + a * (1.0 - s))
        db = g * a * s
        tl.store(d_ptr + row * 2 * H + cols, da.to(d_ptr.dtype.element_ty), mask=m)
        tl.store(d_ptr + row * 2 * H + H + cols, db.to(d_ptr.dtype.element_ty), mask=m)

    @torch.library.custom_op("ncm::swiglu_fwd", mutates_args=())
    def _swiglu_fwd(h: torch.Tensor) -> torch.Tensor:
        h = h.contiguous()
        H = h.shape[-1] // 2
        o = torch.empty(*h.shape[:-1], H, device=h.device, dtype=h.dtype)
        n = h.numel() // (2 * H)
        BLOCK = min(1024, triton.next_power_of_2(H))
        _swiglu_fwd_k[(n, triton.cdiv(H, BLOCK))](h, o, H, BLOCK=BLOCK)
        return o

    @_swiglu_fwd.register_fake
    def _(h):
        return h.new_empty(*h.shape[:-1], h.shape[-1] // 2)

    @torch.library.custom_op("ncm::swiglu_bwd", mutates_args=())
    def _swiglu_bwd(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h, g = h.contiguous(), g.contiguous()
        H = h.shape[-1] // 2
        d = torch.empty_like(h)
        n = h.numel() // (2 * H)
        BLOCK = min(1024, triton.next_power_of_2(H))
        _swiglu_bwd_k[(n, triton.cdiv(H, BLOCK))](h, g, d, H, BLOCK=BLOCK)
        return d

    @_swiglu_bwd.register_fake
    def _(h, g):
        return torch.empty_like(h)

    def _swiglu_setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def _swiglu_back(ctx, g):
        (h,) = ctx.saved_tensors
        return _swiglu_bwd(h, g)

    _swiglu_fwd.register_autograd(_swiglu_back, setup_context=_swiglu_setup)
    swiglu_triton = _swiglu_fwd
else:
    swiglu_triton = None


def _pick_swiglu():
    """Parity first, then a microbenchmark: Triton wins only if it is both correct and faster."""
    if SWIGLU == "torch" or swiglu_triton is None:
        return swiglu_torch, "torch"
    h = torch.randn(4096, 2 * P["ffn"], device=dev, dtype=AMP_DTYPE, requires_grad=True)
    ref = swiglu_torch(h.float())
    got = swiglu_triton(h)
    ok = torch.allclose(got.float(), ref, atol=3e-2, rtol=3e-2)
    g = torch.randn_like(got)
    g1, = torch.autograd.grad(got, h, g)
    g2, = torch.autograd.grad(swiglu_torch(h.float()), h, g.float())
    ok = ok and torch.allclose(g1.float(), g2.float(), atol=3e-2, rtol=3e-2)
    if not ok:
        print("!! triton swiglu failed parity: using torch")
        return swiglu_torch, "torch"
    if SWIGLU == "triton":
        return swiglu_triton, "triton"
    comp = torch.compile(swiglu_torch)
    def bench(f):
        for _ in range(3):
            f(h).sum().backward()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20):
            f(h).sum().backward()
        torch.cuda.synchronize()
        return (time.time() - t) / 20
    try:
        tt, tc = bench(swiglu_triton), bench(comp)
    except Exception as e:
        print("swiglu bench skipped:", type(e).__name__)
        return swiglu_triton, "triton"
    print(f"swiglu fwd+bwd: triton {tt*1e3:.2f} ms | torch.compile {tc*1e3:.2f} ms")
    return (swiglu_triton, "triton") if tt <= tc * 1.05 else (swiglu_torch, "torch")

if IS_CUDA and HAS_TRITON and SHARD_MODE != "encode":      # encode-only runs skip the GPU benchmark and compile
    swiglu, SWIGLU_IMPL = _pick_swiglu()
else:
    swiglu, SWIGLU_IMPL = swiglu_torch, "torch"
print("swiglu kernel:", SWIGLU_IMPL)


# ---- chunked, checkpointed head + cross-entropy ----
def _ce_chunk(h, W, b, tgt):
    logits = F.linear(h, W, b).float()
    lse = torch.logsumexp(logits, dim=-1)
    ce = lse - logits.gather(-1, tgt[:, None]).squeeze(-1)
    return torch.stack([ce, lse])


def fused_ce(h, W, b, tgt, chunk=16384, ckpt=True):
    """h [N, d], tgt [N] -> [2, N] = (cross-entropy, logsumexp). Logits live one chunk at a time."""
    from torch.utils.checkpoint import checkpoint
    outs = []
    for i in range(0, h.shape[0], chunk):
        args = (h[i:i + chunk], W, b, tgt[i:i + chunk])
        if ckpt and torch.is_grad_enabled():
            outs.append(checkpoint(_ce_chunk, *args, use_reentrant=False))
        else:
            outs.append(_ce_chunk(*args))
    return torch.cat(outs, dim=1)
