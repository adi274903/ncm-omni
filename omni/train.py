"""Training loop for the omni generator. One loop, two arms (block-diagonal vs AR mask).

Deliberately plain: the run is small enough that a custom loop beats fighting a Trainer, and it
keeps train/val/eval on the exact same batch contract the producer wrote.

python -m omni.train cfg.json            # real run
python -m omni.train cfg.json --smoke 10  # 10 steps, print s/step + projected cost, save nothing
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from tqdm.auto import tqdm

from omni.data import EvalSet, ShardStream, collate, schedule_from_meta, to_device
from omni.model import OmniConfig, OmniNCM, omni_loss, omni_loss_stats
from omni.schedule import MODALITIES, OmniSchedule


def fast_math():
    """Turn on the fast paths that are actually available on this box.

    On CUDA the matmuls already go to cuBLAS and the attention to SDPA/flash - there is no
    separate library to call. What is NOT on by default: TF32 for fp32 matmuls (A100/H100/L40S
    only) and cuDNN benchmarking. Setting them is worth 10-40% on Ampere+ and costs nothing.
    A JAX port would re-buy the same cuBLAS kernels for a week of work and lose flex_attention,
    so the lever is kernels-per-step, not framework.
    """
    if not torch.cuda.is_available():
        return {}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:                                          # noqa: BLE001
        pass
    p = torch.cuda.get_device_properties(0)
    return dict(device=p.name, sm=f"{p.major}.{p.minor}", mem_gb=round(p.total_memory / 2**30, 1),
                tf32=True)


def adamw(params, cfg):
    """Fused AdamW when the build has it: one kernel over the whole parameter list instead of
    ~4 python-driven launches per tensor. At ~90 tensors that is a real slice of the step."""
    kw = dict(lr=cfg.get("lr", 3e-4), betas=(0.9, 0.95), weight_decay=cfg.get("wd", 0.01))
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, fused=True, **kw)
        except (TypeError, RuntimeError, AssertionError) as e:      # pragma: no cover
            print(f"!! fused AdamW unavailable ({e}); using the eager optimiser")
    return torch.optim.AdamW(params, **kw)


def step_flops(P, bs, T, mult=1.0):
    """6*N*D per training step (2N forward + 4N backward) - the MFU convention, no sync."""
    return 6.0 * P * bs * T * mult


def warm_batch(sched, ocfg, B, device):
    """A batch that is legal for a heterogeneous head: every level's target is < that level's
    vocab. `synthetic_batch` draws from the widest codebook, which the 256-way L0 head rejects."""
    g = torch.Generator().manual_seed(0)
    codes, present = {}, {}
    for m in sched.modalities:
        flat = torch.zeros(B, sched.flat_len(m), dtype=torch.long)
        for k in range(sched.K):
            l, off = sched.level_len(m, k), sched.offsets(m)[k]
            flat[:, off:off + l] = torch.randint(1, ocfg.vocab_at(m, k), (B, l), generator=g)
        codes[m] = flat.to(device)
        present[m] = torch.ones(B, dtype=torch.bool, device=device)
    return {"codes": codes, "present": present}


def prepare(model, cfg, sched, bs, dev):
    """-> the callable to use for the forward (compiled when available, else the raw module).

    torch.compile fuses the ~10 small kernels per block (two RMSNorms, RoPE, the SiLU gate, the
    residual adds) and is the only way flex_attention runs as a fused kernel. It is compiled HERE
    on a synthetic batch, for three reasons: a graph break or an unsupported mask costs a warning
    instead of a silent 8x slowdown at step 0; the compile time lands before the timed loop so
    pos/s is not nonsense; and checkpoints are still saved from the raw module, so `_orig_mod.`
    never leaks into a state_dict that omni/eval.py has to load.
    """
    if not cfg.get("compile", True) or dev != "cuda":
        return model
    # A broken subgraph (older inductor builds mis-codegen flex_attention for tiny shapes) should
    # cost speed on that subgraph, not the whole run. suppress_errors falls back to eager per
    # frame; the try/except below still catches a failure at compile time.
    torch._dynamo.config.suppress_errors = True
    t0, amp = time.time(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True)
    try:
        fwd = torch.compile(model, dynamic=False)
        batch = warm_batch(sched, model.cfg, bs, dev)
        with amp:
            loss = omni_loss(fwd(batch), batch, sched)
        loss.backward()                                     # compiles the backward graph too
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(), amp:
            fwd(batch)                                      # eval graph, same shapes
        model.train()
        g = torch._dynamo.utils.counters["stats"]
        print(f"torch.compile: ready in {time.time() - t0:.0f}s "
              f"({g.get('unique_graphs', 0)} graphs, {g.get('frames_compiled', 0)} frames "
              f"compiled, {g.get('frames_total', 0)} seen)")
        return fwd
    except Exception as e:                                      # noqa: BLE001
        print(f"!! torch.compile failed ({type(e).__name__}: {e}); training eager")
        model.zero_grad(set_to_none=True)
        model.train()
        return model


def build(cfg):
    sch = cfg["sched"]
    sched = OmniSchedule(base_len=sch["base_len"], K=sch["K"], R=sch["R"],
                         null_code=sch.get("null_code", 0),
                         modalities=tuple(sch.get("modalities") or MODALITIES))
    # `vocab` may be {m: int} (old), {m: [V_0..V_{K-1}]} (heterogeneous head) or {m: {k: V}}.
    # OmniConfig normalises it; passing the raw structure through keeps the cfgs diffable.
    vocab = cfg["vocab"]
    m = cfg["model"]
    ocfg = OmniConfig(sched, vocab, n_embd=m["n_embd"], n_layer=m["n_layer"],
                      n_head=m["n_head"], n_kv_head=m["n_kv_head"], dropout=m.get("dropout", 0.0),
                      qk_norm=m.get("qk_norm", True), mask_mode=m.get("mask_mode", "block"),
                      n_experts=m.get("n_experts", 0), expert_map=m.get("expert_map"))
    return OmniNCM(ocfg), sched


def make_writer(out_dir):
    """TensorBoard over the same run dir. Returns None (with the fix printed) if unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:                                          # noqa: BLE001
        print("!! tensorboard is not installed: losses still go to metrics*.jsonl")
        print("   pip install -q tensorboard   then:  %load_ext tensorboard")
        print(f"   %tensorboard --logdir {os.path.join(out_dir, 'tb')}")
        return None
    d = os.path.join(out_dir, "tb")
    os.makedirs(d, exist_ok=True)
    print(f"tensorboard: %load_ext tensorboard; %tensorboard --logdir {d}")
    return SummaryWriter(d)


def preflight_gate(cfg, writer=None):
    """Runs test C before any training. Aborts unless it passed (override: preflight.allow_fail)."""
    pf = cfg.get("preflight") or {}
    if not pf.get("enabled", True):
        print("preflight: disabled in cfg (preflight.enabled=false)")
        return True
    from omni.preflight import run_preflight
    res = run_preflight(cfg, writer=writer)
    if not res["passed"] and not pf.get("allow_fail", False):
        print("\nPREFLIGHT FAILED - refusing to start training.")
        print("  the head/routing does not learn, so more steps cannot fix it.")
        print("  set preflight.allow_fail=true only to debug the trainer itself.")
        raise SystemExit(3)
    return bool(res["passed"])


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flop_params(model):
    """Parameters a token actually flows through, which is what the TFLOPS column must divide by.

    With modality-routed experts `n_params` counts every expert while a synthetic batch only
    routes to a few, so using it inflates the number by the ratio of experts to active experts.
    """
    if getattr(model, "moe", False):
        from omni.moe import active_params
        return active_params(model)
    return n_params(model)


def lr_at(step, cfg):
    warm = max(1, int(cfg.get("warmup", 100)))
    peak = cfg.get("lr", 3e-4)
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, cfg["steps"] - warm)
    return 0.05 * peak + 0.95 * peak * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, evals, sched, device, batch_size=8):
    """Per (modality, level) loss + teacher-forced accuracy, plus a scalar val loss per group."""
    model.eval()
    agg, tot_loss, tot_n = {}, 0.0, 0
    for group, ds in evals.items():
        for i in range(0, len(ds), batch_size):
            rows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            batch = to_device(collate(rows, sched), device)
            logits = model(batch)
            for k in range(sched.K):
                for m in sched.modalities:
                    l, off = sched.level_len(m, k), sched.offsets(m)[k]
                    tg = batch["codes"][m][:, off:off + l]
                    keep = (tg != sched.null_code) & batch["present"][m].unsqueeze(1)
                    if not keep.any():
                        continue
                    lg = logits[(m, k)]
                    ce = torch.nn.functional.cross_entropy(
                        lg.reshape(-1, lg.size(-1))[keep.reshape(-1)],
                        tg.reshape(-1)[keep.reshape(-1)], reduction="sum")
                    acc = (lg.argmax(-1) == tg)[keep].float().mean()
                    key = (m, k)
                    cur = agg.setdefault(key, {"ce": 0.0, "acc": 0.0, "n": 0, "groups": {}})
                    cur["ce"] += float(ce)
                    cur["acc"] += float(acc) * int(keep.sum())
                    cur["n"] += int(keep.sum())
                    g = cur["groups"].setdefault(group, {"acc": 0.0, "n": 0})
                    g["acc"] += float(acc) * int(keep.sum())
                    g["n"] += int(keep.sum())
    model.train()
    out = {}
    for (m, k), v in agg.items():
        out[f"{m}_L{k}"] = dict(ce=v["ce"] / max(1, v["n"]), acc=v["acc"] / max(1, v["n"]),
                                n=v["n"],
                                by_group={g: d["acc"] / max(1, d["n"]) for g, d in v["groups"].items()})
    return out


def run(cfg, smoke=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = make_writer(out_dir)
    preflight_gate(cfg, writer)
    model, sched = build(cfg)
    model.to(dev).train()
    P = n_params(model)
    Pf = flop_params(model)
    T = sched.total_len
    bs = cfg.get("batch", 4)
    if dev == "cuda":
        print("fast math:", fast_math())
    print(f"params {P/1e6:.2f}M | packed positions/sample {T} | modalities {list(sched.modalities)}")
    print(sched.describe())

    evals = {}
    for group, path in (cfg.get("eval_files") or {}).items():
        if os.path.exists(path):
            evals[group] = EvalSet(path)
    print("eval sets:", {g: len(d) for g, d in evals.items()})

    stream = ShardStream(cfg["shard_dir"], consume=cfg.get("consume", False),
                         start_sample=cfg.get("start_sample", 0),
                         producer_log=cfg.get("producer_log"),
                         timeout_s=cfg.get("stream_timeout", 600))
    opt = adamw(model.parameters(), cfg)
    steps = smoke or cfg["steps"]
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics = open(os.path.join(out_dir, "metrics.jsonl"), "a")
    use_bf16 = dev == "cuda"
    fwd = prepare(model, cfg, sched, bs, dev)

    it = iter(stream)
    t0, toks = time.time(), 0
    pbar = tqdm(range(steps), desc="train", unit="step", disable=not sys.stderr.isatty())
    for step in pbar:
        rows = []
        while len(rows) < bs:
            try:
                rows.append(next(it))
            except StopIteration:
                print("stream ended before the step budget: stopping")
                steps = step
                break
        if len(rows) < bs:
            break
        batch = to_device(collate(rows, sched), dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, {**cfg, "steps": steps})
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = fwd(batch)
            loss = omni_loss(logits, batch, sched, label_smoothing=cfg.get("label_smoothing", 0.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        opt.step()
        toks += bs * T
        el = time.time() - t0
        rec = dict(step=step, loss=float(loss.detach()), grad_norm=float(gn.detach()),
                   lr=opt.param_groups[0]["lr"],
                   samples_per_s=round(toks / T / el, 2),   # toks/T is samples, not steps
                   steps_per_s=round((step + 1) / el, 2),
                   pos_per_s=round(toks / el, 1),
                   tflops=round(step_flops(Pf, bs, T) * (step + 1) / el / 1e12, 2))
        if writer is not None and (step % max(1, cfg.get("log_every", 10)) == 0
                                   or step == steps - 1):
            with torch.no_grad():
                _, st = omni_loss_stats(logits, batch, sched)
            writer.add_scalar("train/loss", float(loss.detach()), step)
            writer.add_scalar("train/grad_norm", float(gn), step)
            writer.add_scalar("train/lr", rec["lr"], step)
            writer.add_scalar("data/pos_per_s", rec["pos_per_s"], step)
            writer.add_scalar("perf/tflops", rec["tflops"], step)
            for (mm, kk), v in st.items():
                writer.add_scalar(f"train/ce/{mm}_L{kk}", v["ce"], step)
        pbar.set_postfix(loss=f"{rec['loss']:.4f}", pos_s=rec["pos_per_s"],
                         tflops=rec["tflops"])
        if step % cfg.get("log_every", 10) == 0 or step == steps - 1:
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            pbar.write(json.dumps(rec))
        if evals and cfg.get("eval_every") and (step + 1) % cfg["eval_every"] == 0:
            ev = evaluate(fwd, evals, sched, dev, bs)
            metrics.write(json.dumps(dict(step=step, val=ev)) + "\n")
            metrics.flush()
            fin = {k: round(v["acc"], 4) for k, v in ev.items()}
            print("val acc:", fin)
        if cfg.get("save_every") and (step + 1) % cfg["save_every"] == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                       os.path.join(ckpt_dir, f"step{step+1}.pt"))
    dt = time.time() - t0
    if smoke:
        sps = dt / max(1, steps)
        print(f"SMOKE {steps} steps | {sps:.2f} s/step | "
              f"{toks/dt:.0f} pos/s | {step_flops(Pf, bs, T) / sps / 1e12:.1f} TFLOPS | "
              f"projected {sps*cfg['steps']/3600:.2f} h for {cfg['steps']} steps")
        print(f"cost at $2.13/h on-demand H100: ${sps*cfg['steps']/3600*2.13:.0f}  "
              f"(at $1.00/h spot: ${sps*cfg['steps']/3600*1.0:.0f})")
        return
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": steps - 1},
               os.path.join(out_dir, "final.pt"))
    print("saved", os.path.join(out_dir, "final.pt"))


def run_duplex(cfg, smoke=None):
    """Both arms, one stream, one pass. The batch is built once and stepped by every arm, so the
    A/B cannot drift: same rows, same order, same packing. Costs one model's memory per arm and
    replaces produce-then-train-two-times."""
    import copy

    from omni.livestream import iter_rows

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = make_writer(out_dir)
    preflight_gate(cfg, writer)
    arms = cfg.get("arms") or ["strict", "ar"]
    steps = smoke or cfg["steps"]
    bs = cfg.get("batch", 4)
    use_bf16 = dev == "cuda"
    if dev == "cuda":
        print("fast math:", fast_math())

    models, opts, fwds, sched, arm_cfg = {}, {}, {}, None, {}
    for arm in arms:
        c = copy.deepcopy(cfg)
        c["model"]["mask_mode"] = arm
        # The checkpoint must carry ITS arm's cfg: eval names an arm from mask_mode, and the
        # shared run cfg says "block" (the default), which relabels both arms as the legacy
        # leaky arm and lets the second one silently overwrite the first.
        arm_cfg[arm] = c
        m, s = build(c)
        m.to(dev).train()
        models[arm] = m
        opts[arm] = adamw(m.parameters(), cfg)
        sched = sched or s
        print(f"{arm:6s} params {n_params(m)/1e6:.2f}M")
    T = sched.total_len
    for arm in arms:                      # after sched exists: prepare() needs the packed length
        fwds[arm] = prepare(models[arm], cfg, sched, bs, dev)
    per_step = sum(step_flops(flop_params(models[a]), bs, T) for a in arms)
    print(f"packed positions/sample {T} | arms {arms} | single pass over the live stream")

    evals = {}
    for group, path in (cfg.get("eval_files") or {}).items():
        if os.path.exists(path):
            evals[group] = EvalSet(path)
    print("eval sets:", {g: len(d) for g, d in evals.items()} or "none yet")

    live_cfg = cfg.get("live_cfg")
    from omni.livestream import iter_rows
    if live_cfg:
        rows = iter_rows(live_cfg, holdout=live_cfg.get("holdout"),
                         eval_dir=live_cfg.get("out_dir"))
        print("data: live Hub stream (one pass, ~1.5 samples/s measured - a kill test only)")
    else:
        rows = iter(ShardStream(cfg["shard_dir"], consume=cfg.get("consume", False),
                                start_sample=cfg.get("start_sample", 0),
                                producer_log=cfg.get("producer_log"),
                                timeout_s=cfg.get("stream_timeout", 600)))
        print(f"data: shards in {cfg['shard_dir']} (both arms read the same shards)")
    metrics = {a: open(os.path.join(out_dir, f"metrics_{a}.jsonl"), "a") for a in arms}
    it = iter(rows)
    t0, toks = time.time(), 0
    pbar = tqdm(range(steps), desc=f"duplex {arms}", unit="step",
              disable=not sys.stderr.isatty())
    for step in pbar:
        batch_rows = []
        while len(batch_rows) < bs:
            try:
                batch_rows.append(next(it))
            except StopIteration:
                print("stream ended: stopping")
                steps = step
                break
        if len(batch_rows) < bs:
            break
        batch = to_device(collate(batch_rows, sched), dev)
        rec: dict = {"step": step}
        for arm in arms:
            for g in opts[arm].param_groups:
                g["lr"] = lr_at(step, {**cfg, "steps": steps})
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = fwds[arm](batch)
                loss = omni_loss(logits, batch, sched,
                                 label_smoothing=cfg.get("label_smoothing", 0.0))
            opts[arm].zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(models[arm].parameters(), cfg.get("clip", 1.0))
            opts[arm].step()
            rec[arm] = round(float(loss.detach()), 5)
            rec[f"{arm}_gn"] = round(float(gn.detach()), 3)
            if writer is not None and (step % max(1, cfg.get("log_every", 10)) == 0
                                       or step == steps - 1):
                with torch.no_grad():
                    _, st = omni_loss_stats(logits, batch, sched)
                writer.add_scalar(f"{arm}/loss", float(loss.detach()), step)
                writer.add_scalar(f"{arm}/grad_norm", float(gn), step)
                for (mm, kk), v in st.items():
                    writer.add_scalar(f"{arm}/ce/{mm}_L{kk}", v["ce"], step)
        toks += bs * T
        rec["pos_per_s"] = round(toks / (time.time() - t0), 1)
        rec["tflops"] = round(per_step * (step + 1) / (time.time() - t0) / 1e12, 2)
        if writer is not None:
            writer.add_scalar("data/pos_per_s", rec["pos_per_s"], step)
            writer.add_scalar("perf/tflops", rec["tflops"], step)
        pbar.set_postfix({a: f"{rec[a]:.4f}" for a in arms},
                         pos_s=rec["pos_per_s"], tflops=rec["tflops"])
        if step % max(1, cfg.get("log_every", 10)) == 0 or step == steps - 1:
            pbar.write(json.dumps(rec))
            for arm in arms:
                metrics[arm].write(json.dumps(rec) + "\n")
                metrics[arm].flush()
        if evals and cfg.get("eval_every") and (step + 1) % cfg["eval_every"] == 0:
            for arm in arms:
                # evaluate() returns {m_Lk: {ce, acc, n}}; val loss is the token-weighted CE.
                ev = evaluate(models[arm], evals, sched, dev, bs)
                tot_n = sum(v["n"] for v in ev.values())
                vl = sum(v["ce"] * v["n"] for v in ev.values()) / max(1, tot_n)
                pbar.write(f"  val {arm} @{step+1}: loss {vl:.4f}")
                for key, v in ev.items():
                    metrics[arm].write(json.dumps(dict(step=step, val=True, level=key,
                                                       loss=round(v["ce"], 5),
                                                       acc=round(v["acc"], 5),
                                                       n=v["n"])) + "\n")
                metrics[arm].write(json.dumps(dict(step=step, val=True, loss=round(vl, 5))) + "\n")
                metrics[arm].flush()
                if writer is not None:
                    writer.add_scalar(f"{arm}/val_loss", vl, step)
                    for key, v in ev.items():
                        writer.add_scalar(f"{arm}/val_{key}", v["ce"], step)
        if cfg.get("save_every") and (step + 1) % cfg["save_every"] == 0:
            for arm in arms:
                d = os.path.join(out_dir, f"train_{arm}")
                os.makedirs(d, exist_ok=True)
                torch.save({"model": models[arm].state_dict(), "cfg": arm_cfg[arm], "step": step},
                           os.path.join(d, f"step{step+1}.pt"))

    # The live-stream generator holds 20 open Hub readers and daemon pump threads. Never closing
    # it leaves the interpreter hanging after the last step (process alive, cell never returns),
    # with the streams still attached. Closing runs its finally and stops the pumps.
    close = getattr(rows, "close", None)
    if close:
        try:
            close()
        except Exception:                                          # noqa: BLE001
            pass

    dt = time.time() - t0
    for arm in arms:
        d = os.path.join(out_dir, f"train_{arm}")
        os.makedirs(d, exist_ok=True)
        torch.save({"model": models[arm].state_dict(), "cfg": arm_cfg[arm], "step": steps - 1},
                   os.path.join(d, "final.pt"))
        print("saved", os.path.join(d, "final.pt"))
    print(f"DUPLEX {steps} steps | {dt/max(1, steps):.2f} s/step for {len(arms)} arms "
          f"| {toks/dt:.0f} pos/s | {steps*bs:,} samples seen once")
    if live_cfg:
        have = sorted(f for f in os.listdir(live_cfg["out_dir"]) if f.startswith("eval_"))
        print("eval blocks:", have or "NONE")
    # Colab: closing the generator stops the pump threads, but the Hub readers and hf_xet keep
    # file descriptors open and the interpreter never returns, so the cell sits RUNNING after
    # "DUPLEX ... steps" with every checkpoint already on disk. This must come after the
    # save flush and before any early return (--smoke used to return here and hang the same
    # way). Everything is on disk by now; leave hard rather than hang a finished run.
    if writer is not None:
        writer.flush()
        writer.close()
    if cfg.get("hard_exit", True):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    if cfg.get("arms"):
        run_duplex(cfg, smoke=a.smoke or None)
    else:
        run(cfg, smoke=a.smoke or None)


if __name__ == "__main__":
    main()
