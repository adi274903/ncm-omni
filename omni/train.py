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
import time

import torch

from omni.data import EvalSet, ShardStream, collate, schedule_from_meta, to_device
from omni.model import OmniConfig, OmniNCM, omni_loss
from omni.schedule import MODALITIES, OmniSchedule


def build(cfg):
    sch = cfg["sched"]
    sched = OmniSchedule(base_len=sch["base_len"], K=sch["K"], R=sch["R"],
                         null_code=sch.get("null_code", 0))
    vocab = {m: int(cfg["vocab"][m]) for m in sched.modalities}
    m = cfg["model"]
    ocfg = OmniConfig(sched, vocab, n_embd=m["n_embd"], n_layer=m["n_layer"],
                      n_head=m["n_head"], n_kv_head=m["n_kv_head"], dropout=m.get("dropout", 0.0),
                      qk_norm=m.get("qk_norm", True), mask_mode=m.get("mask_mode", "block"))
    return OmniNCM(ocfg), sched


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lr_at(step, cfg):
    warm = max(1, int(cfg.get("warmup", 100)))
    peak = cfg.get("lr", 3e-4)
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, cfg["steps"] - warm)
    return 0.05 * peak + 0.95 * peak * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, evals, sched, device, batch_size=8):
    """Per (modality, level) loss and teacher-forced accuracy on the held-out shards."""
    model.eval()
    agg = {}
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
    model, sched = build(cfg)
    model.to(dev).train()
    P = n_params(model)
    T = sched.total_len
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                            betas=(0.9, 0.95), weight_decay=cfg.get("wd", 0.01))
    steps = smoke or cfg["steps"]
    bs = cfg.get("batch", 4)
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics = open(os.path.join(out_dir, "metrics.jsonl"), "a")
    use_bf16 = dev == "cuda"

    it = iter(stream)
    t0, toks = time.time(), 0
    for step in range(steps):
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
            logits = model(batch)
            loss = omni_loss(logits, batch, sched, label_smoothing=cfg.get("label_smoothing", 0.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        opt.step()
        toks += bs * T
        rec = dict(step=step, loss=float(loss.detach()), grad_norm=float(gn.detach()),
                   lr=opt.param_groups[0]["lr"],
                   samples_per_s=round((step + 1) / (time.time() - t0), 2),
                   pos_per_s=round(toks / (time.time() - t0), 1))
        if step % cfg.get("log_every", 10) == 0 or step == steps - 1:
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            print(json.dumps(rec))
        if evals and cfg.get("eval_every") and (step + 1) % cfg["eval_every"] == 0:
            ev = evaluate(model, evals, sched, dev, bs)
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
              f"{toks/dt:.0f} pos/s | projected {sps*cfg['steps']/3600:.2f} h for {cfg['steps']} steps")
        print(f"cost at $2.13/h on-demand H100: ${sps*cfg['steps']/3600*2.13:.0f}  "
              f"(at $1.00/h spot: ${sps*cfg['steps']/3600*1.0:.0f})")
        return
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": steps - 1},
               os.path.join(out_dir, "final.pt"))
    print("saved", os.path.join(out_dir, "final.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    run(cfg, smoke=a.smoke or None)


if __name__ == "__main__":
    main()
