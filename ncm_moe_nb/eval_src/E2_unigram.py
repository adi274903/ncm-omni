# %% [markdown]
# ## E2. Rebuild the constant-predictor baseline
#
# During training the baseline is the running unigram of the training targets per (modality, level). That histogram is not saved in
# the checkpoint, so it is rebuilt here from a fresh sample of the **training stream**, encoded with the restored codecs. Without it
# the report would fall back to the held-out entropy, an optimistic baseline. It is an approximation of the training unigram (a
# sample, with equal row-type shares), so baselines can differ slightly from those printed during training.

# %%
UNI_ROWS = _knob("UNI_ROWS", 1500)
UNI.clear()
_rows = collect_rows(None, {r: UNI_ROWS for r in ROW_TYPES}, 1, PARTS, WORKERS, seed=SEED + 99, even=True)
for r in ROW_TYPES:
    rr = [x for x in _rows if x["t"] == r]
    m, B = ROW_MOD[r], MICRO_B[r]
    for i in range(0, len(rr), B):
        mb = encode_rows(r, rr[i:i + B], codecs)
        for k, c in enumerate(mb["codes"]):
            UNI[(m, k)] = UNI.get((m, k), 0) + torch.bincount(c.reshape(-1), minlength=SPEC.V).float()
print("training-stream unigram rebuilt from", {r: len([x for x in _rows if x['t'] == r]) for r in ROW_TYPES}, "rows per row type")
del _rows
gc.collect()
