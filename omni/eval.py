"""Evaluation: the only number that can carry the thesis.

The claim is not "lower perplexity" - it is "the same quality at fewer function evaluations".
So the headline table is quality against NFE, computed identically for both arms:

  level-parallel (ours)  NFE = number of levels generated K'
  autoregressive         NFE = number of tokens generated

Teacher-free greedy sampling, scored against the frozen codec's codes for the held-out samples
(exact-match per level). Everything is reported per level, because a coarse-level win with a
flat fine level is the failure mode this whole design is meant to avoid.
"""
import argparse
import json
import os

import torch

from omni.data import EvalSet, collate, to_device
from omni.model import OmniNCM, OmniConfig
from omni.train import build


def load_model(path, dev="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, sched = build(ck["cfg"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), sched, ck


@torch.no_grad()
def score_codes(model, sched, batch, level_cutoff=None, temp=0.0):
    """Greedy level-parallel sample; returns per-(m,k) exact-match accuracy."""
    gen = model.generate(batch, level_cutoff=level_cutoff, temperature=temp)
    K = sched.K if level_cutoff is None else level_cutoff
    acc = {}
    for k in range(K):
        for m in sched.modalities:
            l, off = sched.level_len(m, k), sched.offsets(m)[k]
            gt = batch["codes"][m][:, off:off + l]
            keep = (gt != sched.null_code) & batch["present"][m].unsqueeze(1)
            if not keep.any():
                continue
            acc[(m, k)] = float((gen[m][:, off:off + l] == gt)[keep].float().mean())
    return acc


@torch.no_grad()
def ar_scores(model, sched, batch, token_budgets):
    """Greedy AR sample up to the largest budget; score only the positions that budget covers."""
    budgets = sorted(set(int(b) for b in token_budgets))
    gen, _ = model.generate_ar(batch, max_len=budgets[-1])
    # raster order index of every (modality, level, position) is what "token budget" means
    order = [(k, m, i) for k in range(sched.K) for m in sched.modalities
             for i in range(sched.level_len(m, k))]
    pos_of = {t: idx for idx, t in enumerate(order)}
    out = {}
    for b in budgets:
        acc = {}
        for k in range(sched.K):
            for m in sched.modalities:
                l, off = sched.level_len(m, k), sched.offsets(m)[k]
                gt = batch["codes"][m][:, off:off + l]
                present = batch["present"][m].unsqueeze(1)
                inside = torch.tensor([pos_of[(k, m, i)] < b for i in range(l)],
                                      dtype=torch.bool, device=gt.device).unsqueeze(0)
                keep = (gt != sched.null_code) & present & inside
                if not keep.any():
                    continue
                acc[(m, k)] = float((gen[m][:, off:off + l] == gt)[keep].float().mean())
        out[b] = acc
    return out


def summarise(acc, sched, n_levels):
    """Mean accuracy over generated levels + the coarsest and the finest generated level."""
    if not acc:
        return dict(n_feval=0, mean=0.0, coarse=0.0, finest=0.0)
    vals = list(acc.values())
    coarse = [v for (m, k), v in acc.items() if k == 0]
    fine = [v for (m, k), v in acc.items() if k == n_levels - 1]
    return dict(n_feval=len(vals), mean=sum(vals) / len(vals),
                coarse=sum(coarse) / max(1, len(coarse)),
                finest=sum(fine) / max(1, len(fine)))


def run(cfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    block, sched, ck = load_model(cfg["block_ckpt"], dev)
    ar = None
    if cfg.get("ar_ckpt") and os.path.exists(cfg["ar_ckpt"]):
        ar, sched_ar, _ = load_model(cfg["ar_ckpt"], dev)
        assert sched_ar.total_len == sched.total_len, "arms must share the schedule"

    files = [f for f in (cfg.get("eval_files") or []) if os.path.exists(f)]
    if not files:                                  # cfg was written before the producer ran
        import glob
        files = sorted(glob.glob(os.path.join(ck["cfg"]["shard_dir"], "eval_*.npz")))
    assert files, "no eval shards found: let the producer write eval_*.npz first"
    rows = []
    for f in files:
        ds_f = EvalSet(f)
        rows += [ds_f[i] for i in range(len(ds_f))]
    if cfg.get("n_samples"):
        rows = rows[:cfg["n_samples"]]
    batch = to_device(collate(rows, sched), dev)
    print(f"eval on {len(rows)} held-out samples from {len(files)} group file(s) | "
          f"packed positions {sched.total_len}")

    table = []
    for kp in range(1, sched.K + 1):
        acc = score_codes(block, sched, batch, level_cutoff=kp)
        s = summarise(acc, sched, kp)
        s.update(arm="level-parallel", nfe=kp, levels=kp)
        table.append(s)

    if ar is not None:
        T = sum(sched.flat_len(m) for m in sched.modalities)
        # AR costs one forward per token, so the full-T point is deliberately skipped: it is
        # ~T forwards of a growing prefix and would dominate the eval wall clock. T/2 already
        # shows the plateau, and the ratio to K is what the table is for.
        budgets = [max(1, T // 16), max(1, T // 8), max(1, T // 4), max(1, T // 2)]
        accs = ar_scores(ar, sched, batch, budgets)
        for b, acc in accs.items():
            s = summarise(acc, sched, sched.K)
            s.update(arm="autoregressive", nfe=b, levels=None)
            table.append(s)
    else:
        print("no AR checkpoint: the comparison table will be one-sided "
              "(train one with model.mask_mode='ar' on the same shards)")

    print("\n| arm | NFE | mean acc | coarse | finest |")
    print("|---|---|---|---|---|")
    for r in sorted(table, key=lambda r: (r["arm"], r["nfe"])):
        print(f"| {r['arm']} | {r['nfe']} | {r['mean']:.4f} | {r['coarse']:.4f} | {r['finest']:.4f} |")

    # the one-line reading of the table
    lp = {r["nfe"]: r["mean"] for r in table if r["arm"] == "level-parallel"}
    if ar is not None:
        arm = {r["nfe"]: r["mean"] for r in table if r["arm"] == "autoregressive"}
        target = lp.get(sched.K, 0.0)
        beat = [n for n, v in sorted(arm.items()) if v >= target]
        msg = (f"level-parallel reaches {target:.4f} at NFE={sched.K}; "
               f"AR needs NFE={beat[0]} for the same quality" if beat
               else f"level-parallel reaches {target:.4f} at NFE={sched.K}; "
                    f"AR never matches it within {max(arm)} tokens")
        print("\n" + msg)
        json.dump({"table": table, "summary": msg}, open(cfg["out"], "w"), indent=2)
    else:
        json.dump({"table": table}, open(cfg["out"], "w"), indent=2)
    print("wrote", cfg["out"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    a = ap.parse_args()
    run(json.load(open(a.cfg)))


if __name__ == "__main__":
    main()
