"""The reader must walk every shard, not the first one forever.

Regression for a bug that produced two whole training runs on a single shard: with consume=False
nothing is deleted, so `files[0]` is the same file on every pass, and the trainer silently trains
on 60-odd rows while the log looks completely normal. Nothing in the loss curve reveals it.
"""
import glob
import os
import shutil
import sys
import tempfile
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omni.data as D

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run", "stream")


def make_dir(n_shards=3):
    """Reuse real shards so meta/schedule parsing is exercised, not stubbed."""
    src = sorted(glob.glob(os.path.join(SRC, "shard_*.npz")))
    if not src:
        print("SKIP: no shards in run/stream - run `python run_omni.py produce --budget 128` first")
        return None
    d = tempfile.mkdtemp()
    for i in range(n_shards):
        shutil.copy(src[i % len(src)], os.path.join(d, f"shard_{i:06d}.npz"))
    return d


def main():
    d = make_dir()
    if not d:
        return

    reads = {}
    orig = D.read_npz

    def spy(path):
        reads[os.path.basename(path)] = reads.get(os.path.basename(path), 0) + 1
        return orig(path)

    D.read_npz = spy
    try:
        st = D.ShardStream(d, consume=False)
        rows = list(itertools.islice(iter(st), 400))
    finally:
        D.read_npz = orig

    assert len(reads) == 3, f"only {sorted(reads)} were read; the reader is stuck on one shard"
    assert len(rows) == 400, f"yielded {len(rows)} rows"
    print(f"[no-consume] read {len(reads)} distinct shards, counts {sorted(reads.values())}")

    # consume=True keeps its old contract: pop the oldest, stop when the producer is done.
    d2 = make_dir()
    open(os.path.join(d2, "DONE"), "w").write("1")
    st = D.ShardStream(d2, consume=True)
    n = len(list(iter(st)))
    assert n > 0, "consume=True yielded nothing"
    assert not glob.glob(os.path.join(d2, "shard_*.npz")), "consume=True left shards behind"
    print(f"[consume]    yielded {n} rows then stopped at DONE, {len(os.listdir(d2)) - 1} files left")
    print("SHARD STREAM OK: every shard is read, epochs wrap")


if __name__ == "__main__":
    main()
