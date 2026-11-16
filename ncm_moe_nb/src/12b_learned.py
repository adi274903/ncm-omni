# %% [markdown]
# ## 14b. Did it learn? (the evidence the verdict cell cannot give)
#
# The verdict cell asks whether the loss fell. A falling loss can still be memorisation, a prefix the model ignores, or a codec whose
# codes are unpredictable. These checks use the held-out set (never trained on) and the codec checks from section 7d:
#
# 1. **Held-out gain per modality and level.** CE against the constant predictor (the training unigram of that level), plus top-1
#    accuracy against the unigram's. A level counts as *learned* when its CE gain is above 0.02 nats (accuracy is shown, not used: a skewed
#    unigram such as silence makes top-1 a poor test); a modality needs a positive finest-level gain and a positive slot-weighted gain. Next to it, the **probe ceiling** from the codec
#    check: how much a small net could win at that level from the previous level alone. A model gain far below a positive probe gain
#    points at the model / data / training length; a probe gain of ~0 points at the codec.
# 2. **Does it use the prefix?** CE with the true caption / prompt / transcript against a shuffled one and a blank one. If the three
#    are equal the conditioning is ignored.
# 3. **Does each level use the level before it?** CE with the real previous-level codes against random ones.
# 4. **Train against held-out CE.** A held-out CE far above the training CE is memorisation.
#
# The final line per modality is `LEARNED`, `PARTIAL` or `NOT LEARNING`, with the reason.

# %%
LEARNED = {}


@torch.no_grad()
def _ce_levels(mb, cond, codes_in):
    """held-out CE of every level of one micro-batch, with `cond` / `codes_in` as the (possibly corrupted) inputs."""
    r = mb["rtype"]
    with _amp():
        h = net.hidden(r, cond, codes_in)
    return [float(F.cross_entropy(net.level_logits(r, h, k).reshape(-1, SPEC.V).float(), mb["codes"][k].reshape(-1)))
            for k in range(LAYOUTS[r].K)]


def _wavg(items):
    tot = sum(n for _, n in items)
    return [sum(c[k] * n for c, n in items) / tot for k in range(len(items[0][0]))]


def learned_report():
    net.eval()
    v = evaluate()
    mods = {}
    for key, d in v.items():
        m, k = key.split("_L")
        mods.setdefault(m, {})[int(k)] = d
    # ---- 1. held-out gain over the constant predictor
    print("1. held-out CE vs the constant-predictor baseline (nats), top-1 accuracy vs the baseline's, and the codec probe ceiling")
    summary = {}
    for m, ks in mods.items():
        print(f"  {m}: level |   CE  | baseline |  gain  |  acc  | base acc | probe ceiling")
        gains, ws, n_ok = [], [], 0
        for k in sorted(ks):
            d = ks[k]
            gain = d["base"] - d["ce"]
            u = UNI.get((m, k))
            bacc = float(u.max() / u.sum()) if u is not None and float(u.sum()) > 0 else float("nan")
            pr = PROBE.get((m, k)) if "PROBE" in globals() else None
            ceil = f"{pr['gain_bits'] * math.log(2):+.3f}" if pr else "  n/a"
            ok = gain > 0.02                                                  # CE is the stable metric; top-1 is printed but a skewed unigram (silence) makes it a poor test
            n_ok += ok; gains.append(gain); ws.append(SPEC.lens(m)[k])
            print(f"      L{k}   {d['ce']:6.3f}  {d['base']:7.3f}   {gain:+.3f}  {d['acc']:.3f}   {bacc:.3f}    {ceil}   {'learned' if ok else ''}")
        summary[m] = dict(levels=len(ks), n_ok=n_ok, gain_finest=gains[-1], gain_mean=float(np.mean(gains)),
                          gain_w=float(np.average(gains, weights=ws)),         # weighted by slots: the finest level carries most of the live codes
                          ce=float(np.mean([ks[k]["ce"] for k in ks])))
    steps_done = max([h["step"] for h in HIST if "loss" in h] or [0])
    if steps_done < 200:
        print(f"\nNOTE: only {steps_done} optimizer steps. The model starts at CE ~ ln(257) = 5.55, above the low-entropy baselines of the coarse levels,\n"
              "and needs hundreds of steps to cross them: read a run this short as a pipeline check, not as a verdict.")
    # ---- 2. the prefix
    print("\n2. does it use the prefix? held-out CE (mean over levels) with the true prefix, a shuffled one and a blank one")
    pref, uses_prefix = {}, {}
    # 0.01 nats printed "uses the prefix" for image and text in the 0.3B-code run when the true
    # reading of the same numbers was that they ignored it. 0.05 is the honest bar.
    PREFIX_MIN = 0.05
    for mb in VAL:
        if mb["cond"] is None:
            continue
        n = len(mb["src"])
        t, s, b = (_ce_levels(mb, c, mb["codes"]) for c in (mb["cond"], mb["cond"].roll(1, 0), torch.ones_like(mb["cond"])))
        pref.setdefault(mb["rtype"], []).append(((t, s, b), n))
    for r, items in pref.items():
        T, S, B_ = (_wavg([(x[i], n) for x, n in items]) for i in range(3))
        delta = [s - t for s, t in zip(S, T)]
        uses_prefix[r] = max(delta) > PREFIX_MIN
        print(f"  {r:6s} true {np.mean(T):.3f} | shuffled {np.mean(S):.3f} | blank {np.mean(B_):.3f} | shuffled - true per level {[round(x, 3) for x in delta]}"
              f"  -> {'uses the prefix' if uses_prefix[r] else 'IGNORES the prefix'}")
    # ---- 3. the previous level
    print("\n3. does each level use the level before it? held-out CE with the real previous-level codes vs random ones (per level k >= 1)")
    g = torch.Generator(device=dev).manual_seed(0)
    ctx, uses_ctx = {}, {}
    for mb in VAL:
        r, n = mb["rtype"], len(mb["src"])
        real = _ce_levels(mb, mb["cond"], mb["codes"])
        rand = []
        for k in range(1, LAYOUTS[r].K):
            c2 = list(mb["codes"])
            c2[k - 1] = torch.randint(1, 257, c2[k - 1].shape, device=dev, generator=g)
            rand.append(_ce_levels(mb, mb["cond"], c2)[k])
        ctx.setdefault(r, []).append(((real[1:], rand), n))
    for r, items in ctx.items():
        Rl, Rn = (_wavg([(x[i], n) for x, n in items]) for i in range(2))
        delta = [b - a for a, b in zip(Rl, Rn)]
        uses_ctx[r] = max(delta) > 0.02
        print(f"  {r:6s} random - real per level 1..{len(delta)}: {[round(x, 3) for x in delta]}  -> {'uses the coarser levels' if uses_ctx[r] else 'IGNORES the coarser levels'}")
    # ---- 4. train vs held-out
    print("\n4. training CE vs held-out CE (mean over levels; a large gap is memorisation)")
    tr = [h for h in HIST if "ce" in h][-5:]
    gap = {}
    for m, s in summary.items():
        t_ce = float(np.mean([np.mean([x for k, x in h["ce"].items() if k.startswith(m + "_")]) for h in tr])) if tr else float("nan")
        gap[m] = s["ce"] - t_ce
        print(f"  {m:6s} train {t_ce:.3f} | held-out {s['ce']:.3f} | gap {gap[m]:+.3f}" + ("  !! overfitting" if gap[m] > 0.5 else ""))
    # ---- verdict per modality
    print()
    for m, s in summary.items():
        rts = [r for r in ROW_TYPES if ROW_MOD[r] == m]
        ctx_ok = any(uses_ctx.get(r, False) for r in rts)
        pre_ok = all(uses_prefix.get(r, True) for r in rts)
        learned = s["gain_finest"] > 0.02 and s["gain_w"] > 0.02
        reasons = []
        if not learned:
            reasons.append(f"finest-level gain {s['gain_finest']:+.3f} nats, slot-weighted gain {s['gain_w']:+.3f}, {s['n_ok']}/{s['levels']} levels beat the baseline")
            pr = [PROBE[(m, k)]["gain_bits"] for k in range(1, len(SPEC.lens(m))) if (m, k) in PROBE] if "PROBE" in globals() else []
            if pr and max(pr) < 0.05:
                reasons.append("the codec probe also finds ~no structure: suspect the codec")
            elif pr:
                reasons.append("the codec probe finds structure the model has not yet used: suspect training length / data mix")
        if not ctx_ok:
            reasons.append("ignores the coarser levels")
        if not pre_ok:
            reasons.append("ignores the prefix")
        if gap.get(m, 0) > 0.5:
            reasons.append("overfitting")
        LEARNED[m] = "LEARNED" if learned and ctx_ok and pre_ok and gap.get(m, 0) <= 0.5 else ("PARTIAL" if learned or s["n_ok"] else "NOT LEARNING")
        print(f"{m:6s} {LEARNED[m]:13s} {'; '.join(reasons) if reasons else 'beats the baseline at the finest level, uses the prefix and the coarser levels'}")
    n_ok = sum(x == "LEARNED" for x in LEARNED.values())
    print(f"\nOVERALL: {n_ok}/{len(LEARNED)} modalities LEARNED  {LEARNED}")
    net.train()


if not META_ONLY:
    learned_report()
