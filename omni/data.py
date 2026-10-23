"""Consumer side: shard directory -> packed multimodal batches.

Reads what producer.py wrote, in order, deleting each shard once consumed (the producer keeps
MAX_AHEAD_SHARDS ahead, so the disk footprint stays bounded). If the producer died, the stream
raises with the tail of its log attached instead of hanging forever.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from omni.schedule import MODALITIES, OmniSchedule


def read_npz(path):
    z = np.load(path, allow_pickle=False)
    raw = z["meta"]
    meta = json.loads(raw.item() if getattr(raw, "shape", ()) == () else str(raw))
    return z, meta


def schedule_from_meta(meta: dict) -> OmniSchedule:
    return OmniSchedule(base_len=meta["base_len"], K=meta["K"], R=meta["R"],
                        null_code=meta.get("null_code", 0))


def rows_from_npz(z, meta, mods=MODALITIES):
    n = len(z[f"{mods[0]}_present"]) if f"{mods[0]}_present" in z else 0
    for i in range(n):
        yield {"codes": {m: z[m][i] for m in mods if m in z},
               "present": {m: bool(z[f"{m}_present"][i]) for m in mods if f"{m}_present" in z}}


class ShardStream(IterableDataset):
    """Infinite-ish sampler over the producer's rolling shards."""

    def __init__(self, shard_dir, consume=False, start_sample=0, poll=0.25,
                 producer_log=None, timeout_s=None):
        # consume=False by default: shards are the only copy of the data, and the two arms of the
        # experiment must read the same rows. Deleting them after arm 1 leaves arm 2 with a
        # truncated stream - which silently shortens its step budget instead of failing loudly.
        self.dir, self.consume, self.start = shard_dir, consume, start_sample
        self.poll, self.producer_log, self.timeout_s = poll, producer_log, timeout_s

    def _shards(self):
        return sorted(glob.glob(os.path.join(self.dir, "shard_*.npz")))

    def _tail(self, n=20):
        if not self.producer_log or not os.path.exists(self.producer_log):
            return "(producer log unavailable)"
        with open(self.producer_log) as f:
            return "".join(f.readlines()[-n:])

    def __iter__(self):
        seen, t0, i = 0, time.time(), 0
        while True:
            files = self._shards()
            if files:
                # consume=True: the list shrinks as we delete, so index 0 is always next.
                # consume=False: nothing is deleted, so walking index 0 forever re-reads the
                # same shard - which is a silent single-shard corpus, not a data bug you can see
                # in the loss. Walk the sorted list and wrap for the next epoch.
                path = files[0] if self.consume else files[i % len(files)]
                try:
                    z, meta = read_npz(path)
                except Exception:                        # half-written from the producer
                    time.sleep(self.poll)
                    continue
                for row in rows_from_npz(z, meta):
                    if seen < self.start:                # resume: skip without yielding
                        seen += 1
                        continue
                    seen += 1
                    yield row
                if self.consume:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                else:
                    i += 1
                continue
            if os.path.exists(os.path.join(self.dir, "DONE")):
                return                                   # producer finished and nothing is left
            if self.timeout_s and time.time() - t0 > self.timeout_s:
                raise TimeoutError(f"no shards after {self.timeout_s}s\n{self._tail()}")
            time.sleep(self.poll)


class EvalSet(Dataset):
    def __init__(self, npz_path):
        z, self.meta = read_npz(npz_path)
        self.rows = list(rows_from_npz(z, self.meta))
        self.sched = schedule_from_meta(self.meta)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(rows, sched, mods=None):
    """List of sample dicts -> {"codes": {m: int64 [B, flat_len(m)]}, "present": {m: bool [B]}}.

    Widths always come from the schedule, never from the rows: a shard written by a codec that
    disagreed with the schedule must fail loudly here rather than produce a ragged batch.
    """
    mods = tuple(mods or sched.modalities)
    B = len(rows)
    codes, present = {}, {}
    for m in mods:
        flat = sched.flat_len(m)
        arr = np.full((B, flat), sched.null_code, dtype=np.int64)
        pres = np.zeros(B, dtype=bool)
        for i, r in enumerate(rows):
            c = r["codes"].get(m)
            if c is not None and len(c) == flat:
                arr[i] = c
                pres[i] = bool(r["present"].get(m, False))
        codes[m] = torch.from_numpy(arr)
        present[m] = torch.from_numpy(pres)
    return {"codes": codes, "present": present}


def to_device(batch, device):
    return {"codes": {m: v.to(device, non_blocking=True) for m, v in batch["codes"].items()},
            "present": {m: v.to(device, non_blocking=True) for m, v in batch["present"].items()}}


def synthetic_batch(sched: OmniSchedule, vocab, B=2, seed=0, device="cpu", present_all=True):
    """A batch that needs no producer: for selfcheck and for the overfit-one-batch smoke test."""
    g = torch.Generator().manual_seed(seed)
    codes, present = {}, {}
    for m in sched.modalities:
        flat = torch.randint(1, vocab[m], (B, sched.flat_len(m)), generator=g)
        codes[m] = flat.to(device)
        present[m] = torch.ones(B, dtype=torch.bool, device=device)
    if not present_all:
        present[sched.modalities[-1]] = torch.zeros(B, dtype=torch.bool, device=device)
    return {"codes": codes, "present": present}
