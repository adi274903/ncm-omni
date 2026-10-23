"""Rows straight off the Hub, with no shard corpus in between.

`omni.producer` writes shards because separate processes - one per arm - need to read identical
data. A duplex run does not: both arms consume the same batch, in the same process, in the same
order, so the stream *is* the control variable and 337 MB of npz files never have to exist.

Pulls are concurrent. A single-threaded loop over 20 Hub streams stalls completely whenever one
source is slow - an audio decode or a video download blocks `Source.batch` and nothing else gets
a turn. Text alone measures 4.9 docs/s, so that stall, not bandwidth, is what makes a naive
single-process stream look dead. The producer hid it by running 8 processes.

The trade is reproducibility. A shard corpus can be re-read after a crash; a stream is one-pass.
That is right for a kill test (short, one comparison) and wrong for a long pretraining run.
"""
import os
import queue
import random
import threading

from omni.producer import (GROUPS, SOURCES, Source, atomic_npz, encode_sample, log as plog,
                           make_codecs, schedule_meta, stack_samples)
from omni.schedule import OmniSchedule


def _sched_of(pcfg):
    """The schedule lives under `sched` in a train cfg and flat in a produce cfg; accept both."""
    s = pcfg.get("sched") or {}

    def pick(k, d=None):
        for src in (pcfg, s):
            v = src.get(k)
            if v is not None:
                return v
        return d

    return OmniSchedule(base_len=pick("base_len"), K=pick("K"), R=pick("R"),
                        null_code=pick("null_code", 0))


def _pump(src, out, stop, batch_docs):
    """One thread per source: pull, hand over, repeat. Daemon so a hung source cannot block exit."""
    while not stop.is_set():
        got = src.batch(batch_docs)
        if not got:
            out.put((src.name, None))
            return
        for parts in got:
            if stop.is_set():
                return
            out.put((src.name, parts))


def iter_rows(pcfg, holdout=None, eval_dir=None, verbose=True, max_queue=512):
    """Yield encoded samples, interleaved by the configured source weights.

    If `holdout` is given ({group: n}), the first n rows of each group are diverted, encoded as
    eval_{group}.npz under `eval_dir`, and never yielded - the held-out set comes off the same
    stream the arms read, so it cannot drift from training.
    """
    sched = _sched_of(pcfg)
    V = pcfg["codebook_size"]
    codecs = make_codecs(sched, V, pcfg.get("prefer_real", True))
    meta = schedule_meta(sched, codecs, V)
    rng = random.Random(pcfg.get("seed", 3407))

    mix = {k: float(v) for k, v in pcfg["mix"].items() if float(v) > 0}
    total_w = sum(mix.values())
    frac = {k: v / total_w for k, v in mix.items()}

    holdout = dict(holdout or {})
    held = {g: [] for g in set(GROUPS.values())}
    if holdout and eval_dir:
        os.makedirs(eval_dir, exist_ok=True)

    q, stop = queue.Queue(maxsize=max_queue), threading.Event()
    live = {}
    for name in mix:
        src = Source(name, *SOURCES[name][:2])
        live[name] = src
        threading.Thread(target=_pump, args=(src, q, stop, int(pcfg.get("batch_docs", 1))),
                         daemon=True).start()
        if verbose:
            plog(f"[live] {name} pumping (sharded={src.sharded})")

    pending = dict.fromkeys(live, 0)             # parts queued but not yet drawn
    pulled = 0
    try:
        while True:
            name, parts = q.get()
            if parts is None:
                live.pop(name, None)
                plog(f"[live] {name} exhausted ({len(live)} left)")
                if not live:
                    return
                continue
            pulled += 1
            codes, present = encode_sample(codecs, sched, parts)
            if not any(present.values()):
                continue
            row = {"codes": codes, "present": present}
            group = GROUPS[name]
            want = holdout.get(group, 0)
            if want and len(held[group]) < want:
                held[group].append(row)
                if len(held[group]) == want and eval_dir:
                    p = os.path.join(eval_dir, f"eval_{group}.npz")
                    atomic_npz(p, stack_samples(held[group], sched), meta)
                    plog(f"[eval] {group}: {want} held out -> {p}")
                continue
            yield row
            if verbose and pulled % 500 == 0:
                plog(f"[live] {pulled} rows pulled | sources {len(live)} | held "
                     f"{ {g: len(v) for g, v in held.items()} }")
    finally:
        stop.set()
