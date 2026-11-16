# %% [markdown]
# ### Guard: this notebook must never train
# Stops here if the first cell (the `%env` lines) was not run, because otherwise the training cell further down would start a run.

# %%
assert EVAL_ONLY, "NCM_EVAL_ONLY is not set: run the first cell (the %env lines) and restart the runtime before running anything else. This notebook only evaluates."
print("EVAL_ONLY is on: no training will run.")
