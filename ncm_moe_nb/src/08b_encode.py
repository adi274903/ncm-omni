# %% [markdown]
# ## 7. Encode the dataset to shards (only when `SHARD_MODE="encode"`)
#
# Streams the whole dataset for the chosen budget (`SHARD_CODES_B`, else `TOKENS`, else the tier: T3 = 50B live codes,
# about 60 GB), encodes it with the frozen codebooks fitted above and stores it per row type under `SHARD_DIR`. No model
# step runs, so it is limited by the Hub feed alone. It is resumable (run the cell again after an interruption). When it
# finishes, the build, budget, correctness, training, verdict and generation cells skip themselves; set `SHARD_MODE="read"`
# and run the notebook again to train from the stored shards.

# %%
if SHARD_MODE == "encode":
    ENCODED = encode_shards()
    print("encode-only run finished: set SHARD_MODE='read' and run the notebook again to train from the shards."
          if ENCODED else "encode-only run stopped early: run this cell again to continue.")

# %% [markdown]
# ### 7b. Bundle the shards into one archive (to copy to Drive yourself)
#
# Runs after the encode cell when `SHARD_MODE="encode"` and prints how long it took.

# %%
if SHARD_MODE == "encode" and os.path.isdir(SHARD_DIR):
    ARCHIVE = os.path.join(os.path.dirname(os.path.abspath(SHARD_DIR)), os.path.basename(SHARD_DIR.rstrip("/")) + "_" + TIER + ".tar.gz")
    t_tar = time.time()
    subprocess.run(["tar", "-czf", ARCHIVE, "-C", os.path.dirname(os.path.abspath(SHARD_DIR)), os.path.basename(SHARD_DIR.rstrip("/"))],
                   check=True, env={**os.environ, "GZIP": "-1"})
    print(f"tar.gz done in {(time.time() - t_tar) / 60:.1f} min ({time.time() - t_tar:.0f} s) | {os.path.getsize(ARCHIVE) / 2**30:.2f} GiB | {ARCHIVE}")
