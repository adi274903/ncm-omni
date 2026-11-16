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
import time

from collections import deque

from omni.producer import (GROUPS, SOURCES, Source, atomic_npz, encode_sample, log as plog,
                           make_codecs, schedule_meta, stack_samples)
from omni.schedule import OmniSchedule


class _Weighted:
    """Deficit round-robin over sources, so the yielded mix follows `frac` and not the pumps' speed.

    The queue is one FIFO of whatever finished first, and text costs far less to encode than audio
    (no decode, no mel pyramid), so a plain `q.get()` hands back a text-dominated stream: measured on
    the Hub, the first ~600 rows of a run contained no image at all and the first ~4250 no audio.
    `frac` was computed and then never used, so the achieved mix was a function of throughput - a
    458-step run (3664 samples) never reached the first audio row.

    A source may run ahead; its surplus is buffered up to `cap` rows and the oldest dropped beyond
    that (there is always more where it came from, and dropping text is cheap). The source with the
    largest deficit against its target share is served next, so the long-run mix converges on `frac`.
    """
    def __init__(self, frac, cap=64):
        self.frac = {k: float(v) for k, v in frac.items()}
        self.cap = int(cap)
        self.buf = {k: deque() for k in self.frac}
        self.emitted = dict.fromkeys(self.frac, 0)
        self.dropped = dict.fromkeys(self.frac, 0)
        self.cool = set()                    # owed a slot, but it made us wait: stop blocking on it
        self.total = 0

    def push(self, name, item):
        self.cool.discard(name)               # it delivered: it may be waited on again
        buf = self.buf.get(name)
        if buf is None:                       # a source outside the mix: accept it, weight it 0
            self.buf[name] = buf = deque()
            self.frac[name] = 0.0
            self.emitted[name] = 0
            self.dropped[name] = 0
        buf.append(item)
        if len(buf) > self.cap:
            buf.popleft()
            self.dropped[name] += 1

    def next(self):
        """The source most owed a slot right now - buffered or not, and possibly not yet delivered.

        Choosing on deficit alone is the whole point. Choosing among *buffered* sources is FIFO in
        disguise: the fast source is always the one with something in hand, so it always wins.
        """
        best, best_def = None, None
        n = max(1, self.total)
        for name in self.frac:
            if self.frac[name] <= 0.0 or name in self.cool:   # retired, unmixed, or just timed out
                continue
            d = self.frac[name] * n - self.emitted[name]
            if best_def is None or d > best_def:
                best, best_def = name, d
        return best

    def best_ready(self):
        """The most owed source that already has a row buffered - the patience fallback."""
        best, best_def = None, None
        n = max(1, self.total)
        for name, buf in self.buf.items():
            if not buf or self.frac.get(name, 0.0) <= 0.0:
                continue
            d = self.frac[name] * n - self.emitted[name]
            if best_def is None or d > best_def:
                best, best_def = name, d
        return best

    def ready(self, name):
        return bool(self.buf.get(name))

    def take(self, name):
        self.emitted[name] += 1
        self.total += 1
        return self.buf[name].popleft()

    def retire(self, name):
        """A pump is gone: stop waiting on it, but still serve whatever it already delivered."""
        if name in self.frac:
            self.frac[name] = 0.0
            self.emitted[name] = 0
        self.cool.discard(name)

    def cool_down(self, name):
        """It made us wait past `patience`; hand its slot on until it actually delivers again.

        Without this a source that is merely slow costs one row per patience window forever, which
        stalls the stream far worse than the mixing bug did.
        """
        self.cool.add(name)

    def shares(self):
        n = max(1, self.total)
        return {k: round(v / n, 3) for k, v in sorted(self.emitted.items()) if v}


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
    codecs = make_codecs(sched, V, pcfg.get("prefer_real", True), pcfg.get("codec_tier", "auto"))
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
    live, threads = {}, []
    for name in mix:
        src = Source(name, *SOURCES[name][:2])
        live[name] = src
        t = threading.Thread(target=_pump, args=(src, q, stop, int(pcfg.get("batch_docs", 1))),
                             daemon=True)
        t.start()
        threads.append(t)
        if verbose:
            plog(f"[live] {name} pumping (sharded={src.sharded})")

    # Weights the draw order, not the arrival order. `patience` bounds how long a source that is owed
    # a slot may hold everything up: past it the most-owed *available* source is served instead, so a
    # slow pump throttles the stream while a dead one cannot hang it.
    wq = _Weighted(frac)
    patience = float(pcfg.get("mix_patience", 3.0))
    pulled = 0
    try:
        while True:
            # The clock, not a counter of empty reads. Text almost never lets the queue go empty, so
            # anything that only counted timeouts would wait forever on the first source that is owed
            # a slot but has not delivered yet - which is exactly the startup stall (19 pumps up, no
            # rows out, for 150 s).
            t_draw = time.monotonic()
            while True:
                owed = wq.next()
                if owed is None:
                    return                      # every source retired: nothing left to wait for
                if wq.ready(owed):
                    name = owed
                    break
                fallback = wq.best_ready()
                if fallback is not None and time.monotonic() - t_draw >= patience:
                    wq.cool_down(owed)          # it kept us waiting: let the next draw skip it
                    name = fallback
                    break
                if not any(t.is_alive() for t in threads) and q.empty():
                    if fallback is not None:
                        name = fallback
                        break
                    return                      # every pump dead: exhausted, or it raised
                try:
                    src, parts = q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if parts is None:
                    live.pop(src, None)
                    wq.retire(src)
                    plog(f"[live] {src} exhausted ({len(live)} left)")
                    continue
                wq.push(src, parts)
            parts = wq.take(name)
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
                plog(f"[live] {pulled} rows | sources {len(live)} | mix {wq.shares()} "
                     f"| dropped {sum(wq.dropped.values())} "
                     f"| held { {g: len(v) for g, v in held.items()} }")
    finally:
        stop.set()
