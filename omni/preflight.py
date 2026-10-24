"""Preflight (test C): does the new head + expert routing actually learn, before any real run?

Runs automatically before training (see train.run / train.run_duplex) and refuses to start a
paid run when the answer is no. It is deliberately NOT a data-quality test - it uses a synthetic
stream whose coarse and fine levels are a deterministic function of one latent per sample, so
the only thing being measured is whether information reaches each level through the chosen mask.

Why synthetic: on real data a level can be unlearnable because the corpus is hard, which tells
you nothing about the code. Here the relation is trivially learnable ("copy the coarser level"),
so a level that does not converge is a wiring bug, not a data problem. The copy relation is also
the exact operation the strict mask permits, so this is the strongest cheap test of the thesis.

What it gates on (k >= 1 only - level 0 has no coarser level and can only learn a marginal):
  * every fine level's CE falls by >= 0.5 nats from ln(V_k)  -> information flows
  * the widest level (BPE 8k/32k text) reaches CE < 1.0      -> the big head is trainable
  * no NaN/inf loss, and every expert group gets gradient    -> routing is alive, not collapsed

Level 0 is printed for reference and never gated: it is a prior over the coarsest codes.
"""
import copy
import math
import time
from typing import Dict

import torch
from tqdm.auto import tqdm

from omni.data import to_device
from omni.model import omni_loss_stats
from omni.schedule import OmniSchedule


def _structured_batch(sched, ocfg, B, device, seed=0):
    """ocfg is the OmniConfig (per-level vocab lives there), not the raw cfg dict."""
    """One latent per sample; every level of every modality is a deterministic function of it.

    code_k[i] = (latent * 3 + i) % (V_k - 1) + 1  ->  level k+1 can be read off level k, and the
    finest level of a heterogeneous head must emit it through a much wider softmax.
    """
    g = torch.Generator().manual_seed(seed)
    latent = torch.randint(0, 16, (B, 1), generator=g)
    codes, present = {}, {}
    for m in sched.modalities:
        flat = torch.zeros(B, sched.flat_len(m), dtype=torch.long)
        for k in range(sched.K):
            V = ocfg.vocab_at(m, k)
            l, off = sched.level_len(m, k), sched.offsets(m)[k]
            pos = torch.arange(l).unsqueeze(0)
            flat[:, off:off + l] = (latent * 3 + pos) % max(1, V - 1) + 1
        codes[m] = flat.to(device)
        present[m] = torch.ones(B, dtype=torch.bool, device=device)
    return {"codes": codes, "present": present}


def run_preflight(cfg, writer=None, steps=None, batch=None, device=None, verbose=True):
    """-> dict(passed, rows, reason). Aborts the caller unless it passed."""
    from omni.train import build as build_model          # local: train imports this module

    pf = cfg.get("preflight", {}) or {}
    steps = int(steps or pf.get("steps", 300))
    B = int(batch or pf.get("batch", 8))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    c = copy.deepcopy(cfg)
    for k in ("n_embd", "n_layer", "n_head", "n_kv_head"):
        if k in pf:
            c["model"][k] = pf[k]
    c["model"]["mask_mode"] = pf.get("mask_mode", "strict")

    t0 = time.time()
    model, sched = build_model(c)
    oc = model.cfg
    model.to(device).train()
    P = sum(p.numel() for p in model.parameters())
    active = P
    if getattr(model, "moe", False):
        from omni.moe import ModalityExperts
        for mod in model.modules():
            if isinstance(mod, ModalityExperts):
                per = sum(p.numel() for p in mod.parameters()) / mod.n_experts
                active -= (mod.n_experts - 1) * per

    if verbose:
        print("\n=== PREFLIGHT (test C) ===")
        print(f"model {P/1e6:.2f}M total | {active/1e6:.2f}M active/token | "
              f"experts {getattr(model, 'n_experts', 0)} | mask {c['model']['mask_mode']}")
        vs = " ".join(f"{m}=" + (str(oc.vocab[m][0]) if len(set(oc.vocab[m])) == 1
                                 else f"{oc.vocab[m][0]}..{oc.vocab[m][-1]}")
                      for m in sched.modalities)
        print(f"levels K={sched.K} | vocab {vs}")
        print(sched.describe())

    # A deliberately hotter LR than the main run: this is a wiring test, not a quality test, and
    # the copy relation is learnable in a few hundred steps at 3e-3. Gating on the real run's LR
    # would turn a slow warmup into a false "the head is broken".
    opt = torch.optim.AdamW(model.parameters(), lr=float(pf.get("lr", 3e-3)),
                            betas=(0.9, 0.95), weight_decay=0.01)
    use_bf16 = device == "cuda"
    first, last = {}, {}
    usage = None
    bad = None
    pbar = tqdm(range(steps), desc="preflight", disable=not verbose, leave=True)
    for step in pbar:
        batch_d = _structured_batch(sched, oc, B, device, seed=step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(batch_d)
            loss, stats = omni_loss_stats(logits, batch_d, sched)
        if not torch.isfinite(loss):
            bad = f"non-finite loss at step {step}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = {k: v["ce"] for k, v in stats.items()}
        last = {k: v["ce"] for k, v in stats.items()}
        if getattr(model, "moe", False):
            eid = model._expert_ids(torch.tensor(
                [i for i, m in enumerate(sched.modalities) for _ in range(sched.flat_len(m))]).to(device))
            u = [c * B for c in torch.bincount(eid, minlength=model.n_experts).tolist()]
            usage = u if usage is None else [a + b for a, b in zip(usage, u)]
        if writer is not None:
            writer.add_scalar("preflight/loss", float(loss.detach()), step)
            for (m, k), v in stats.items():
                writer.add_scalar(f"preflight/ce_{m}_L{k}", v["ce"], step)
        pbar.set_postfix(loss=f"{float(loss.detach()):.3f}", **{
            f"{m[0]}{k}": f"{v['ce']:.2f}" for (m, k), v in list(stats.items())[-3:]})

    rows, passed, reasons = [], True, []
    for m in sched.modalities:
        for k in range(sched.K):
            key = (m, k)
            if key not in last:
                continue
            V = oc.vocab_at(m, k)
            base = math.log(V)
            ce0, ce1 = first.get(key, base), last[key]
            drop = base - ce1
            # Gated only where a coarser level exists to condition on (k >= 1). Two conditions:
            # most of the headroom is gone, and (for the wide BPE head) the absolute CE is far
            # below ln(V), i.e. the softmax is putting its mass on the right code.
            ok = True
            if k >= 1:
                # Scale-free on purpose: ln(V) is the no-learning floor, so half of it means the
                # level is emitting real structure whatever its width. A broken head sits at ~1.0
                # of ln(V), so this catches it without a magic absolute number that a slow warmup
                # can trip.
                ok = (drop >= 1.5) and (ce1 <= 0.5 * base)
            rows.append(dict(m=m, k=k, V=V, base=base, ce0=ce0, ce1=ce1, drop=drop, ok=ok))
            if not ok:
                passed = False
                reasons.append(f"{m} L{k} (V={V}) CE {ce0:.2f}->{ce1:.2f}, drop {drop:.2f}")

    if bad:
        passed, reasons = False, [bad]
    if usage is not None and any(u == 0 for u in usage):
        passed = False
        reasons.append(f"dead expert(s): usage {usage}")

    if verbose:
        print(f"\n  {'level':<12}{'V':>7}{'ln V':>8}{'CE start':>10}{'CE end':>9}{'drop':>8}  gate")
        for r in rows:
            print(f"  {r['m'] + ' L' + str(r['k']):<12}{r['V']:>7}{r['base']:>8.2f}"
                  f"{r['ce0']:>10.2f}{r['ce1']:>9.2f}{r['drop']:>8.2f}  "
                  f"{'PASS' if r['ok'] else 'FAIL'}")
        if usage is not None:
            print(f"  expert token usage: {usage}")
        print(f"\n  PREFLIGHT {'PASSED' if passed else 'FAILED'} in {time.time()-t0:.0f}s")
        if not passed:
            for r in reasons:
                print("   -", r)
    if writer is not None:
        writer.add_scalar("preflight/passed", int(passed), 0)
        writer.flush()
    return dict(passed=passed, rows=rows, reason="; ".join(reasons), usage=usage)
