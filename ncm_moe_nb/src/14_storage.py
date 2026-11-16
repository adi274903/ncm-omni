# %% [markdown]
# ## 16. What was stored
#
# The HF cache lives in RAM and the only files this run may have written are checkpoints (`CKPT_DIR`) and the encoded
# code shards (`SHARD_DIR`, ~1 byte per code, never raw images or audio). This cell lists any other new file larger than
# 100 KB on disk, checks that the in-RAM Hub cache holds only metadata (< 50 MB), and reports the shard directory.

# %%
def _new_files():
    out = []
    roots = [os.getcwd()]                      # disk only: the RAM cache (/dev/shm) is checked by size below
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in (".git", "__pycache__", ".ipynb_checkpoints", ".venv")]
            if any(d_ and os.path.abspath(dp).startswith(os.path.abspath(d_)) for d_ in (CKPT_DIR, SHARD_DIR)):
                continue
            for f in fn:
                p_ = os.path.join(dp, f)
                try:
                    if os.path.getmtime(p_) >= T_START and os.path.getsize(p_) > 100_000 and not f.endswith((".ipynb", ".py", ".md")):
                        out.append((os.path.getsize(p_), p_))
                except OSError:
                    pass
    return sorted(out, reverse=True)

_nf = _new_files()
ram_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(_RAM_HF) for f in fs) / 2 ** 20 if _RAM_HF else 0
print(f"HF cache in RAM (Hub metadata only): {ram_mb:.1f} MB | new files > 100 KB on disk: {len(_nf)}")
for sz, p_ in _nf[:10]:
    print(f"  {sz/2**20:8.2f} MB  {p_}")
assert not _nf, "this run wrote files other than checkpoints and code shards"
assert ram_mb < 50, f"HF cache in RAM is {ram_mb:.0f} MB: more than metadata was cached"
if USE_SHARDS:
    _nfiles, _nbytes, _rows = shard_stats(SHARD_DIR)
    print(f"shards in {SHARD_DIR}: {_nfiles} files, {_nbytes / 2**30:.2f} GiB, rows {_rows}")
print("OK: no raw data and no dataset cache stored" + (" (only encoded code shards)" if USE_SHARDS else ""))
