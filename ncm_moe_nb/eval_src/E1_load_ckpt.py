# %% [markdown]
# ## E1. Load the checkpoint
#
# The weights go into the model built above; the codecs were already restored by the stage cell. The saved history holds the
# held-out eval from every `eval_every` step of training: printing it shows **where each modality's held-out CE bottomed out**,
# which tells you whether the saved (last) model is also the best one.

# %%
assert EVAL_ONLY, "set NCM_EVAL_ONLY=1 in the first cell"
assert RESUME is not None, (f"no usable checkpoint at {CKPT_PATH or '(CKPT_DIR is empty)'}: set NCM_CKPT_DIR to the training run's folder, "
                            "and keep NCM_DAC_MODEL / NCM_TIER the same as in training")
net.load_state_dict(RESUME["net"])
net.eval()
HIST = RESUME["hist"]
print(f"loaded {CKPT_PATH}: step {RESUME['step']}, codes seen { {k: round(v / 1e9, 3) for k, v in RESUME['seen'].items()} } (B), {len(HIST)} history entries")
RESUME["opt"] = None                                   # the optimizer state is large and not needed here
gc.collect()

vals = [h for h in HIST if "val" in h]
print("\nheld-out CE saved during training (mean over levels; baseline = the training unigram at that time):")
for h in vals:
    mods = {}
    for key, d in h["val"].items():
        mods.setdefault(key.split("_")[0], []).append((d["ce"], d["base"]))
    print(f"  step {h['step']:5d} | " + " | ".join(
        f"{m} {np.mean([a for a, _ in v]):.3f} (gain {np.mean([b - a for a, b in v]):+.3f})" for m, v in mods.items()))
for m in ("text", "image", "audio"):
    ser = [(h["step"], np.mean([d["ce"] for k, d in h["val"].items() if k.startswith(m + "_")])) for h in vals]
    if ser:
        best = min(ser, key=lambda x: x[1])
        print(f"  {m:5s} best held-out CE {best[1]:.3f} at step {best[0]}; last {ser[-1][1]:.3f} at step {ser[-1][0]}"
              + ("   !! the last checkpoint is worse than an earlier eval" if ser[-1][1] > best[1] + 0.02 else ""))
