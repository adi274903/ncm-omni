# %% [markdown]
# ## 14. Did it learn? (verdict from the run's own records)

# %%
def verdict():
    tr = [h for h in HIST if "loss" in h]
    va = [h for h in HIST if "val" in h]
    fails = []
    def need(ok, why):
        if not ok: fails.append(why)
    need(len(tr) >= 1, "no logged steps")
    need(all(math.isfinite(h["loss"]) and math.isfinite(h["gnorm"]) for h in tr), "non-finite loss or grad norm")
    short = len(tr) < 6
    if not short:
        n = max(1, len(tr) // 5)
        early, late = np.median([h["loss"] for h in tr[:n]]), np.median([h["loss"] for h in tr[-n:]])
        drop = 100 * (1 - late / early)
        print(f"loss  median {early:.3f} -> {late:.3f} ({drop:+.1f}%; ln(257) at init = {math.log(257):.3f})")
        need(late <= 0.83 * early, f"loss moved only {drop:.1f}%, want >= 17%")
        per = {}
        for h in tr:
            for k, v in h["ce"].items():
                per.setdefault(k.split("_")[0], {}).setdefault(k, []).append(v)
        moved = 0
        for m, ks in per.items():
            first = np.mean([v[0] for v in ks.values()]); last = np.mean([v[-1] for v in ks.values()])
            print(f"  {m:5s} train CE {first:.3f} -> {last:.3f} ({100*(1-last/first):+.1f}%)")
            moved += (last <= 0.9 * first)
        need(moved >= 2, f"only {moved}/3 modalities' CE fell >= 10%")
    if va:
        v = va[-1]["val"]
        beat = {}
        for key, d in v.items():
            beat.setdefault(key.split("_")[0], []).append(d["base"] - d["ce"])
        gain = {m: float(np.mean(x)) for m, x in beat.items()}
        print("held-out gain over the constant-predictor baseline (nats, mean over levels):", {m: round(g, 3) for m, g in gain.items()})
        if not short:
            need(sum(g > 0.05 for g in gain.values()) >= 2, f"held-out CE does not beat the constant baseline for 2+ modalities: {gain}")
    print()
    if short:
        print(f"VERDICT: TOO SHORT to judge ({len(tr)} logged steps) | pipeline ran end to end, loss finite" if not fails else f"VERDICT: BROKEN {fails}")
    elif fails:
        print("VERDICT:", "NOT LEARNING" if late > 0.95 * early else "PARTIAL")
        for f in fails: print("  -", f)
    else:
        print(f"VERDICT: PASS | loss {drop:.1f}% down | seen {sum(tr[-1]['seen'].values())/1e9:.3f}B live codes")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
        ax[0].plot([h["step"] for h in tr], [h["loss"] for h in tr]); ax[0].set_title("train loss"); ax[0].set_xlabel("step")
        for m in ("text", "image", "audio"):
            ys = [np.mean([v for k, v in h["ce"].items() if k.startswith(m)]) for h in tr]
            ax[1].plot([h["step"] for h in tr], ys, label=m)
        ax[1].legend(); ax[1].set_title("train CE per modality"); ax[1].set_xlabel("step"); plt.tight_layout(); plt.show()
    except Exception:
        pass

if not META_ONLY:
    verdict()
