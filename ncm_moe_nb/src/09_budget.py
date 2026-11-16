# %% [markdown]
# ## 11. Budget: tiers, steps, throughput, ETA (printed before anything expensive runs)
#
# Compute-optimal is ~20 tokens per active parameter (~7B here); capable small models are trained 10-100x
# past that. A "live code" is one predicted (non-prefix) position.
#
# | tier | live codes | text | image | audio | tokens / active param | what it buys |
# |---|---|---|---|---|---|---|
# | T0 smoke | 0.02B | 40% | 35% | 25% | 0.06 | pipeline works, loss moves |
# | T1 pilot | 2B | | | | 5.5 | per-modality CE separates, coarse levels sensible |
# | **T2 foundation-lite (default)** | **20B** | 8B | 7B | 5B | 55 | readable text, recognisable image structure, intelligible TTS |
# | T3 foundation | 50B | 20B | 18B | 12.5B | 140 | still far below the 1-4T tokens published ~360M text LMs saw |
#
# Wall-clock is `6 * N_active * D / sustained FLOPS`: ~3.6 days on an A100 at 45% MFU, ~1.3 days on an
# H100 for T2. The measured numbers below replace these estimates.

# %%
TIERS = {"T0": 0.02, "T1": 2.0, "T2": 20.0, "T3": 50.0}
TOK_B = TOKENS or TIERS[TIER]
if TOKENS and abs(float(TOKENS) - TIERS[TIER]) > 1e-9:
    # The two knobs contradict each other: a truthy TOKENS wins and TIER is only a label.
    # That is how a "TIER=T1" run quietly became a 0.3B-code run (297 steps instead of 1,953).
    print(f"  !! TOKENS={TOKENS}B overrides TIER={TIER} ({TIERS[TIER]}B): this run is {TOK_B:.3f}B live "
          f"codes. Set NCM_TOKENS=0 to take the budget from the tier.")
AVG_LIVE_MICRO = sum(TYPE_P[r] * MICRO_B[r] * LAYOUTS[r].Tt for r in ROW_TYPES)
ACCUM = max(1, round(BATCH_CODES / AVG_LIVE_MICRO))
CODES_PER_STEP = ACCUM * AVG_LIVE_MICRO
STEPS_TOTAL = STEPS or max(1, round(TOK_B * 1e9 / CODES_PER_STEP))
TOK_B = STEPS_TOTAL * CODES_PER_STEP / 1e9

rows_per_code = {r: 1.0 / LAYOUTS[r].Tt for r in ROW_TYPES}
tgt = {"text": TOK_B * CODE_SHARE["text"], "image": TOK_B * CODE_SHARE["image"], "audio": TOK_B * CODE_SHARE["audio"]}
n_text = tgt["text"] * 1e9 / SPEC.text / (1)                       # approx rows of text (live codes per row = ladder total)
print(f"tier {TIER}: {TOK_B:.3f}B live codes = {STEPS_TOTAL:,} optimizer steps x {CODES_PER_STEP:,.0f} codes "
      f"({ACCUM} micro-batches of ~{MICRO_TOKENS} packed positions)")
print(f"tokens per active parameter: {TOK_B*1e9/REPORT['active_max']:.1f} (compute-optimal ~20)")
for m in ("text", "image", "audio"):
    live = LAYOUTS["text0" if m == "text" else m].Tt
    rows = tgt[m] * 1e9 / live
    raw = {"text": f"~{rows * SPEC.text / 1e9:.1f} GB of text", "image": f"~{rows/1e6:.2f}M images",
           "audio": f"<= {rows * SPEC.wave_len / SPEC.sr / 3600:,.0f} h of audio (incl. silence padding)"}[m]
    print(f"  {m:5s} {tgt[m]:7.3f}B live codes = {rows/1e6:8.2f}M rows  ({raw})")
print("row-type mix (rows):", {r: f"{TYPE_P[r]:.3f}" for r in ROW_TYPES})

FLOPS_TOTAL = 6 * REPORT["active_max"] * TOK_B * 1e9 * (sum(TYPE_P[r] * LAYOUTS[r].T for r in ROW_TYPES)
                                                         / sum(TYPE_P[r] * LAYOUTS[r].Tt for r in ROW_TYPES))
print(f"training FLOPs (6 N_active x packed tokens): {FLOPS_TOTAL:.2e}")
for nm, tf in (("A100 @45% MFU", 140), ("H100 bf16 @40%", 400), ("RTX 3070 (measured below)", None)):
    if tf:
        print(f"  {nm:26s} ~{FLOPS_TOTAL / (tf * 1e12) / 3600:7.1f} h")

# %%
def synth(rtype, B):
    lay = LAYOUTS[rtype]
    codes = [torch.randint(1, 257, (B, n), device=dev) for n in lay.lens]
    cond = torch.randint(1, 257, (B, lay.cond), device=dev) if lay.cond else None
    return cond, codes


def _amp():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if (IS_CUDA and BF16) else torch.autocast("cpu", enabled=False)


def bench_compute(iters=3):
    """Measured fwd+bwd time per micro-batch of each row type -> blended live codes/s, packed tokens/s, MFU."""
    net.train()
    t_of = {}
    for r in ROW_TYPES:
        B = MICRO_B[r]
        cond, codes = synth(r, B)
        for i in range(iters + 2):
            if i == 2:
                if IS_CUDA: torch.cuda.synchronize()
                t0 = time.time()
            with _amp():
                loss, _ = loss_fn(r, cond, codes)
            loss.backward()
        if IS_CUDA: torch.cuda.synchronize()
        t_of[r] = (time.time() - t0) / iters
        net.zero_grad(set_to_none=True)
    t_avg = sum(TYPE_P[r] * t_of[r] for r in ROW_TYPES)
    live_s = AVG_LIVE_MICRO / t_avg
    tok_s = sum(TYPE_P[r] * MICRO_B[r] * LAYOUTS[r].T for r in ROW_TYPES) / t_avg
    tflops = 6 * REPORT["active_max"] * tok_s / 1e12
    return t_of, live_s, tok_s, tflops


if not META_ONLY and not EVAL_ONLY:                 # no benchmark in the eval notebook
    t_of, LIVE_PER_S, TOK_PER_S, TFLOPS = bench_compute()
    mfu = f" | MFU {100 * TFLOPS / PEAK_TFLOPS:.0f}% of ~{PEAK_TFLOPS} peak" if PEAK_TFLOPS else ""
    print("seconds per micro-batch:", {r: round(t, 3) for r, t in t_of.items()})
    print(f"compute: {LIVE_PER_S:,.0f} live codes/s | {TOK_PER_S:,.0f} packed tokens/s | {TFLOPS:.1f} TFLOPS{mfu}")
    if IS_CUDA:
        print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB of {VRAM_GB:.0f}")
    eta_h = TOK_B * 1e9 / LIVE_PER_S / 3600
    need_rows = LIVE_PER_S / (sum(TYPE_P[r] * LAYOUTS[r].Tt for r in ROW_TYPES))
    print(f"==> {TOK_B:.3f}B codes would take ~{eta_h:,.1f} h on this device if the feed keeps up; "
          f"the feed must deliver ~{need_rows:,.0f} rows/s ({WORKERS} workers -> ~{need_rows / max(WORKERS, 1):.1f} rows/s each)")
