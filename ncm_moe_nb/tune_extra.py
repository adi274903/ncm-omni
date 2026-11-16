"""Hands-off speed sweep for a new GPU (H100, A100, MI300X, ...). Nothing here changes the model.

    python run_src.py --upto 09 --extra tune_extra.py

run_src.py builds everything up to the compute benchmark (synthetic batches, no data feed), then this file re-runs the
benchmark for each (ATTN, MOE_DISPATCH) pair on the real model and prints a table. Every pair recompiles the step (a few
minutes each), so the sweep takes roughly 15-25 min. Env knobs:

    NCM_MICRO_TOKENS=32768 | 49152     micro-batch size (one value per invocation: it fixes the batch shapes at build time)
    NCM_ATTN=auto                      also build the flex-attention masks, so "flex" joins the sweep
    NCM_TUNE_AUTOTUNE=1                finish with the best pair under COMPILE_MODE=max-autotune-no-cudagraphs (+15-20 min)
    NCM_TUNE_RESID=1                   also try the bf16 residual stream (RESID) on the best pair

Confirm the winner with a short real run (python run_src.py with NCM_STEPS=40) before adopting it: on an RTX PRO 6000 a
per-layer microbenchmark picked the wrong attention mode, so only whole-step numbers are trusted here.
"""
import json, os

_results = []
_attn0, _disp0, _resid0, _loss0 = ATTN, MOE_DISPATCH, RESID, loss_fn


def _bench(label, attn, disp, mode=None, resid=None):
    global ATTN, MOE_DISPATCH, RESID, loss_fn
    ATTN, MOE_DISPATCH = attn, disp
    RESID = resid or _resid0
    use_flex = attn == "flex"
    if use_flex:                                   # the flex path is selected by ATTN="auto" plus net.use_flex
        ATTN = "auto"
    net.use_flex = use_flex
    torch._dynamo.reset()
    if COMPILED:
        loss_fn = torch.compile(net.loss, dynamic=False, mode=mode or COMPILE_MODE)
    if IS_CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        t_of, live_s, tok_s, tflops = bench_compute()
    except Exception as e:                         # a mode that does not work on this GPU must not stop the sweep
        print(f"  {label:44s} FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
        _results.append(dict(label=label, attn=attn, dispatch=disp, resid=RESID, mode=mode or COMPILE_MODE, error=str(e)[:200]))
        net.zero_grad(set_to_none=True)
        return None
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if IS_CUDA else 0.0
    mfu = f" | MFU {100 * tflops / PEAK_TFLOPS:.0f}%" if PEAK_TFLOPS else ""
    print(f"  {label:44s} {live_s:9,.0f} codes/s | {tflops:6.1f} TFLOPS{mfu} | peak {peak:.1f} GiB | {time.time() - t0:.0f}s", flush=True)
    r = dict(label=label, attn=attn, dispatch=disp, resid=RESID, mode=mode or COMPILE_MODE, codes_s=live_s, tflops=tflops, peak_gib=peak)
    _results.append(r)
    return r


print(f"\n=== speed sweep on {GPU_NAME} | micro-batch {MICRO_TOKENS} | torch {torch.__version__} ===")
_attn_modes = ["levels", "sdpa"] + (["flex"] if hasattr(net.lay["audio"], "flex_mask") else [])
for _a in _attn_modes:
    for _d in ("flat", "gather"):
        _bench(f"attn={_a} dispatch={_d}", _a, _d)

_ok = [r for r in _results if "codes_s" in r]
if not _ok:
    raise SystemExit("every configuration failed: see the messages above")
_best = max(_ok, key=lambda r: r["codes_s"])

if os.environ.get("NCM_TUNE_RESID") == "1":
    _bench(f"{_best['label']} resid=bf16", _best["attn"], _best["dispatch"], resid="bf16")
if os.environ.get("NCM_TUNE_AUTOTUNE") == "1":
    _bench(f"{_best['label']} compile=max-autotune", _best["attn"], _best["dispatch"], mode="max-autotune-no-cudagraphs")

_ok = [r for r in _results if "codes_s" in r]
_best = max(_ok, key=lambda r: r["codes_s"])
_base = next((r for r in _ok if r["attn"] == "sdpa" and r["dispatch"] == "gather"), _ok[0])
print(f"\nbest: {_best['label']}: {_best['codes_s']:,.0f} codes/s, {_best['tflops']:.1f} TFLOPS "
      f"({100 * (_best['codes_s'] / _base['codes_s'] - 1):+.1f}% vs {_base['label']})")
print(f"use:  NCM_ATTN={'auto' if _best['attn'] == 'flex' else _best['attn']} NCM_MOE_DISPATCH={_best['dispatch']} NCM_RESID={_best['resid']} "
      f"NCM_COMPILE_MODE={_best['mode']} NCM_MICRO_TOKENS={MICRO_TOKENS}")
_out = os.path.join(WORK_DIR, f"tune_results_{GPU_NAME.replace(' ', '_')}_{MICRO_TOKENS}.json")
try:
    with open(_out, "w") as f:
        json.dump(dict(gpu=GPU_NAME, torch=torch.__version__, micro_tokens=MICRO_TOKENS, results=_results), f, indent=1)
    print("results written to", _out)
except Exception as e:
    print("could not write results:", e)

ATTN, MOE_DISPATCH, RESID, loss_fn = _attn0, _disp0, _resid0, _loss0
