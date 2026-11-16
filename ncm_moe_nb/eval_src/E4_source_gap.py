# %% [markdown]
# ## E4. Which source is memorised? Per-source train versus held-out CE
#
# The mean train-versus-held-out gap hides which data is responsible. This scores fresh rows from the **training partitions** (the
# shards the model trained on; small sources are re-read many times, so these are mostly seen data) and rows from the **held-out
# partition** (part 0, never trained on), teacher-forced, grouped by source. A source whose training-partition CE is far below its
# held-out CE is the one being memorised. Rows are not the exact ones seen in training, so the gap is a lower-confidence estimate.

# %%
GAP_N = {"audio": _knob("GAP_AUDIO", 800), "image": _knob("GAP_IMAGE", 400), "text0": _knob("GAP_TEXT", 600)}


@torch.no_grad()
def _ce_rows(rtype, rows, bs=16):
    """source -> (mean CE per level over its rows, n rows), teacher-forced."""
    K_ = LAYOUTS[rtype].K
    acc = {}
    net.eval()
    for i in range(0, len(rows), bs):
        mb = encode_rows(rtype, rows[i:i + bs], codecs)
        with _amp():
            h = net.hidden(rtype, mb["cond"], mb["codes"])
        per = []
        for k in range(K_):
            lg = net.level_logits(rtype, h, k).float()
            ce = F.cross_entropy(lg.reshape(-1, SPEC.V), mb["codes"][k].reshape(-1), reduction="none").view(lg.shape[0], -1).mean(1)
            per.append(ce)
        per = torch.stack(per, 1).cpu().numpy()                             # [B, K]
        for j, s in enumerate(mb["src"]):
            a = acc.setdefault(s, [np.zeros(K_), 0])
            a[0] += per[j]; a[1] += 1
    return {s: (v[0] / v[1], v[1]) for s, v in acc.items()}


if not META_ONLY:
    for kind in ("audio", "image", "text0"):
        n = GAP_N[kind]
        held = collect_rows((kind,), {kind: n}, 0, PARTS, 1, seed=12345)
        seen = collect_rows((kind,), {kind: n}, 1, PARTS, WORKERS, seed=SEED + 7)
        h_, t_ = _ce_rows(kind, held), _ce_rows(kind, seen)
        del held, seen
        print(f"\n{kind}: source | rows train/held | CE train | CE held-out | gap | finest-level gap   (CE = mean over levels, nats)")
        for s in sorted(set(h_) | set(t_)):
            if s in h_ and s in t_:
                (ch, nh), (ct, nt) = h_[s], t_[s]
                gap = float(ch.mean() - ct.mean())
                print(f"  {s:10s} {nt:4d}/{nh:<4d}  {ct.mean():7.3f}  {ch.mean():7.3f}  {gap:+.3f}  {float(ch[-1] - ct[-1]):+.3f}" + ("   !! memorised?" if gap > 0.5 else ""))
            else:
                print(f"  {s:10s} only in {'train' if s in t_ else 'held-out'} rows (too few to compare)")
        gc.collect()
