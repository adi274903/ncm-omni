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
    # Dense SDPA: eval is 48 sequences, so a fused kernel buys nothing, and flex's
    # per-level shapes (l_0 != l_5) blow the Dynamo cache and fall back to eager anyway.
    gen = model.generate(batch, level_cutoff=level_cutoff, temperature=temp,
                         use_flex=False)
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
    gen, _ = model.generate_ar(batch, max_len=budgets[-1], use_flex=False)
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


@torch.no_grad()
def cond_scores(model, sched, batch):
    """Per-(m,k) accuracy with the TRUE coarser levels given - one forward, no sampling.

    This is the metric that survives level 0 being unpredictable. Free-running generation needs
    a correct level 0 to condition everything else on, and level 0 is a 1-of-256 guess from
    nothing (val text_L0 ~ 0.01), so the free-running number is dominated by that first coin
    flip. Conditional accuracy asks the question the architecture actually answers: given the
    coarser levels, how much of level k do you reconstruct in ONE pass?
    """
    logits = model(batch, use_flex=False)
    acc = {}
    for k in range(sched.K):
        for m in sched.modalities:
            l, off = sched.level_len(m, k), sched.offsets(m)[k]
            gt = batch["codes"][m][:, off:off + l]
            keep = (gt != sched.null_code) & batch["present"][m].unsqueeze(1)
            if not keep.any():
                continue
            pred = logits[(m, k)].argmax(-1)
            acc[(m, k)] = float((pred == gt)[keep].float().mean())
    return acc


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


ARMS = (("level-parallel", "block_ckpt"),            # legacy: sees its own level, do not quote
        ("level-parallel(strict)", "strict_ckpt"),   # the honest parallel arm
        ("autoregressive", "ar_ckpt"))


def run(cfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Any subset of checkpoints may exist: eval on whatever has been trained, and say what is
    # missing. A missing arm is a message, not a crash - the point is never to be blocked from
    # reading the arm you do have.
    loaded, sched, ck = {}, None, None
    for name, key in ARMS:
        p = cfg.get(key)
        if not (p and os.path.exists(p)):
            continue
        mdl, sch, meta = load_model(p, dev)
        if sched is None:
            sched, ck = sch, meta
        else:
            assert sch.total_len == sched.total_len, "arms must share the schedule"
        # Name the arm from the checkpoint's OWN mask_mode, not from which file it sat in. A
        # pre-shift `block` run parked in run/train_block is the legacy arm, and must never be
        # reported as "level-parallel".
        mm = (meta.get("cfg", {}).get("model", {}) or {}).get("mask_mode", "?")
        if mm == "strict":
            name = "level-parallel(strict)"
        elif mm == "ar":
            name = "autoregressive"
        elif mm == "block":
            name = "level-parallel(LEGACY-leaky)"
        else:
            name = f"unknown({mm})"
        print(f"  {p}  mask_mode={mm}  -> {name}")
        loaded[name] = mdl
    assert loaded, ("no checkpoints found - train at least one arm "
                    "(`run_omni.py train --arm strict`)")
    print("arms found:", ", ".join(loaded))
    block = loaded.get("level-parallel")
    strict = loaded.get("level-parallel(strict)")
    ar = loaded.get("autoregressive")

    files = [f for f in (cfg.get("eval_files") or []) if os.path.exists(f)]
    if not files:                                  # cfg was written before the producer ran
        import glob
        files = sorted(glob.glob(os.path.join(ck["cfg"]["shard_dir"], "eval_*.npz")))
    assert files, "no eval shards found: let the producer write eval_*.npz first"
    rows = []
    for f in files:
        ds_f = EvalSet(f)
        rows += [ds_f[i] for i in range(len(ds_f))]
    if cfg.get("n_samples") and cfg["n_samples"] < len(rows):
        # Round-robin over modalities. A plain `rows[:n]` silently deleted the video group,
        # because the files are read audio, image, text, video and video came last - so "video"
        # columns read as "-" and the smallest modality vanished from every table.
        by = {}
        for r in rows:                             # rows are {"codes", "present"}; key on modality
            m = next((x for x in sched.modalities if r["present"].get(x)), "?")
            by.setdefault(m, []).append(r)
        order = list(by)
        keep, i = [], 0
        while len(keep) < cfg["n_samples"] and any(by[m] for m in order):
            m = order[i % len(order)]
            if by[m]:
                keep.append(by[m].pop(0))
            i += 1
        rows = keep
    batch = to_device(collate(rows, sched), dev)
    print(f"eval on {len(rows)} held-out samples from {len(files)} group file(s) | "
          f"packed positions {sched.total_len}")

    # A constant predictor (always emit the most frequent code of each slot) is the floor any
    # real score has to clear: without it a 0.12 looks like signal when it is below the trivial
    # solution. It must come from the TRAINING distribution - estimated on 48 held-out rows it
    # reads ~0.62, because with 48 rows the modal code of a slot is "most of the slot".
    from collections import Counter
    import glob
    import numpy as np
    counts = {(m, k): Counter() for m in sched.modalities for k in range(sched.K)}
    src = "training shards"
    shards = sorted(glob.glob(os.path.join(ck["cfg"].get("shard_dir", ""), "shard_*.npz")))[:40]
    if shards:
        for f in shards:
            z = np.load(f, allow_pickle=False)
            for m in sched.modalities:
                if m not in z:
                    continue
                # absent modalities are all-NULL padding; counting them makes the baseline the
                # NULL rate, not the code distribution
                keep = z[f"{m}_present"].astype(bool) if f"{m}_present" in z else np.ones(len(z[m]), bool)
                if not keep.any():
                    continue
                for k in range(sched.K):
                    off, l = sched.offsets(m)[k], sched.level_len(m, k)
                    counts[(m, k)].update(z[m][keep][:, off:off + l].reshape(-1).tolist())
    else:                                             # shards gone: fall back, and say so
        src = "held-out rows (INFLATED - no shards on disk)"
        for m in sched.modalities:
            for k in range(sched.K):
                off, l = sched.offsets(m)[k], sched.level_len(m, k)
                counts[(m, k)].update(batch["codes"][m][:, off:off + l].reshape(-1).tolist())
    freq = {key: c.most_common(1)[0][1] / max(sum(c.values()), 1)
            for key, c in counts.items() if c}
    baseline = sum(freq.values()) / max(len(freq), 1)
    # per level, so the conditional table is compared against the right floor at each level
    base_lv = {k: (sum(v for (m, kk), v in freq.items() if kk == k)
                   / max(1, sum(1 for (m, kk) in freq if kk == k)))
               for k in range(sched.K)}
    print(f"constant-predictor baseline from {src} = {baseline:.4f}")

    table = []
    for name, model in loaded.items():               # whatever was actually loaded, labelled
        if "autoregressive" in name:
            continue
        for kp in range(1, sched.K + 1):
            acc = score_codes(model, sched, batch, level_cutoff=kp)
            s = summarise(acc, sched, kp)
            s.update(arm=name, nfe=kp, levels=kp)
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
    if strict is None:
        print("no strict checkpoint: the honest parallel arm has not been trained yet "
              "(`run_omni.py train --arm strict`)")
    if ar is None:
        print("no AR checkpoint: the comparison table will be one-sided "
              "(train one with model.mask_mode='ar' on the same shards)")

    cond = {}
    for name, model in loaded.items():
        cond[name] = cond_scores(model, sched, batch)
    print("\n### conditional: true coarser levels given, one forward per level (no sampling)\n")
    print("| arm | level | text | image | audio | video | mean | constant |")
    print("|---|---|---|---|---|---|---|---|")
    for name, acc in cond.items():
        for k in range(sched.K):
            cells = [acc.get((m, k)) for m in sched.modalities]
            got = [c for c in cells if c is not None]
            row = " | ".join("  -  " if c is None else f"{c:.3f}" for c in cells)
            print(f"| {name} | L{k} | {row} | {sum(got)/max(1,len(got)):.4f} | "
                  f"{base_lv[k]:.3f} |")
    print()
    print("### free-running: sample from level 0, no teacher (the unconditional number)\n")
    print("| arm | NFE | mean acc | coarse | finest |")
    print("|---|---|---|---|---|")
    for r in sorted(table, key=lambda r: (r["arm"], r["nfe"])):
        print(f"| {r['arm']} | {r['nfe']} | {r['mean']:.4f} | {r['coarse']:.4f} | {r['finest']:.4f} |")

    # ---------------------------------------------------------------- verdict
    # Two numbers, and they answer different questions:
    #   conditional  given the true coarser levels, how much of level k comes back in ONE pass
    #   free-running sample from level 0 - the number a paper would quote, and the one that dies
    #                here because level 0 is a 1-of-256 guess from nothing (val text_L0 ~ 0.01),
    #                so every later level is conditioned on a wrong prefix
    mine = ("level-parallel(strict)" if "level-parallel(strict)" in loaded
            else next((n for n in loaded if "level-parallel" in n), "level-parallel"))
    fine = sched.K - 1
    my_cond = sum(cond[mine].get((m, fine), 0.0) for m in sched.modalities
                  if (m, fine) in cond[mine])
    my_cond /= max(1, sum(1 for m in sched.modalities if (m, fine) in cond[mine]))
    ar_cond = None
    if "autoregressive" in cond:
        ar_cond = sum(cond["autoregressive"].get((m, fine), 0.0) for m in sched.modalities
                      if (m, fine) in cond["autoregressive"])
        ar_cond /= max(1, sum(1 for m in sched.modalities if (m, fine) in cond["autoregressive"]))

    free_best = max([0.0] + [r["mean"] for r in table])
    lines = [f"conditional L{fine} ({'one forward' if True else ''}): {mine} {my_cond:.4f}"
             + (f" vs AR {ar_cond:.4f} ({sched.K} forwards vs {sum(sched.flat_len(m) for m in sched.modalities)})"
                if ar_cond is not None else "")
             + f" | constant {base_lv[fine]:.4f}"]
    if my_cond <= base_lv[fine]:
        lines.append(f"VOID: the conditional finest level is at or below the constant predictor "
                     f"({base_lv[fine]:.4f}) - the model has not learned to use the coarser levels.")
    lines.append(f"free-running best = {free_best:.4f} (constant {baseline:.4f})"
                 + ("" if free_best > baseline else
                    " - VOID, and expected: level 0 is unpredictable, so this number measures "
                    "the propagation of the first wrong guess, not the architecture."))
    msg = "\n".join(lines)
    print("\n" + msg)
    json.dump({"table": table, "conditional": {k: {f"{m}_L{kk}": v
                                                   for (m, kk), v in acc.items()}
                                               for k, acc in cond.items()},
               "baseline": baseline, "baseline_by_level": base_lv, "summary": msg},
              open(cfg["out"], "w"), indent=2)
    print("wrote", cfg["out"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    a = ap.parse_args()
    run(json.load(open(a.cfg)))


if __name__ == "__main__":
    main()
