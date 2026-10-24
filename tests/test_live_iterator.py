"""Run the real `iter_rows` against fake pumps, offline. No Hub, no GPU, no torch.

`test_mix_scheduler.py` exercises `_Weighted` in isolation and so cannot catch bugs in the draw loop
that uses it. This one does catch them: the startup stall shipped once - a source that was owed a
slot but had not delivered yet blocked the generator forever, so 19 pumps came up and zero rows came
out for 150 s. The cause was counting `queue.Empty` reads instead of reading a clock; text never lets
the queue go empty, so the counter stayed at zero.
"""
import itertools
import sys
import time

import omni.livestream as L


class FakeSource:
    """A Hub stream stand-in: `delay` before the first row, `rate` rows/s, then `rows` exhausted."""

    def __init__(self, name, delay=0.0, rate=0.0, rows=10 ** 9):
        self.name, self.sharded = name, False
        self.delay, self.rate, self.rows = delay, rate, rows
        self.produced = 0

    def batch(self, n):
        if self.delay:
            time.sleep(self.delay)
            self.delay = 0.0
        if self.rate <= 0:
            time.sleep(3600)                      # never delivers: a hung or throttled pump
            return []
        time.sleep(1.0 / self.rate)
        if self.produced >= self.rows:
            return []
        self.produced += 1
        return [{"name": self.name}]


CFG = {
    "codebook_size": 257,
    "batch_docs": 1,
    "seed": 7,
    "mix_patience": 1.0,
    "sched": {},
}


def patch(sources):
    """Point every Hub-facing name in `iter_rows` at a fake. `sources` maps name -> FakeSource."""
    L.Source = lambda name, *_a, **_k: sources[name]
    L._sched_of = lambda pcfg: None
    L.make_codecs = lambda *a, **k: {}
    L.schedule_meta = lambda *a, **k: {}
    L.encode_sample = lambda codecs, sched, parts: (None, {L.GROUPS[parts["name"]]: True})


def draw(mix, sources, n):
    cfg = dict(CFG, mix=mix)
    gen = L.iter_rows(cfg, verbose=False)
    try:
        return list(itertools.islice(gen, n))
    finally:
        gen.close()                               # runs the finally: stops the pumps


ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


# 1. The shipped stall: one source owed a slot never delivers, but the other keeps the queue busy.
#    Before the clock fix this blocked forever; with patience it must hand the slot on in ~1 s.
srcs = {"text_fineweb": FakeSource("text_fineweb", rate=400.0),
        "image_imagenet": FakeSource("image_imagenet", rate=0.0)}
patch(srcs)
t0 = time.monotonic()
rows = draw({"text_fineweb": 0.5, "image_imagenet": 0.5}, srcs, 50)
dt = time.monotonic() - t0
check("a source that never delivers cannot stall the stream", len(rows) == 50, f"{len(rows)} rows")
check("...and 50 rows arrive in well under the 150 s the stall cost", dt < 15.0, f"{dt:.1f}s")
check("...all of them from the live source",
      all(list(r["present"]) == ["text"] for r in rows))

# 2. Steady state: three pumps, wildly different rates, all live from the start. The mix must follow
#    the configured weights rather than the rates.
srcs = {"text_fineweb": FakeSource("text_fineweb", rate=400.0),
        "image_imagenet": FakeSource("image_imagenet", rate=60.0),
        "audio_libri": FakeSource("audio_libri", rate=25.0)}
patch(srcs)
mix = {"text_fineweb": 0.4, "image_imagenet": 0.4, "audio_libri": 0.2}
rows = draw(mix, srcs, 300)

present_groups = []
for r in rows:
    for g in r["present"]:
        present_groups.append(g)
n = max(1, len(present_groups))
share = {g: round(present_groups.count(g) / n, 3) for g in ("text", "image", "audio")}
check("achieved mix follows the weights, not the pump rates",
      all(abs(share[g] - mix_w) <= 0.08 for g, mix_w in
          (("text", 0.4), ("image", 0.4), ("audio", 0.2))), str(share))
check("audio is not starved by the 16x faster text pump", share["audio"] > 0.12, str(share))

# 3. A source that starts late is still served once it wakes up, and the mix recovers.
srcs = {"text_fineweb": FakeSource("text_fineweb", rate=400.0),
        "image_imagenet": FakeSource("image_imagenet", delay=0.5, rate=60.0),
        "audio_libri": FakeSource("audio_libri", delay=0.5, rate=25.0)}
patch(srcs)
rows = draw({"text_fineweb": 0.4, "image_imagenet": 0.4, "audio_libri": 0.2}, srcs, 40)
late = [next(iter(r["present"])) for r in rows]
check("a late source is picked up, not cooled forever",
      "audio" in late[20:] or late.count("audio") > 0, f"first 40: {late.count('audio')} audio")

# 4. Exhaustion still terminates the generator instead of hanging on the queue.
srcs = {"text_fineweb": FakeSource("text_fineweb", rate=400.0, rows=5)}
patch(srcs)
rows = draw({"text_fineweb": 1.0}, srcs, 50)
check("an exhausted source ends the stream", len(rows) == 5, f"{len(rows)} rows")

print("\n" + ("ALL LIVE-ITERATOR CHECKS PASS" if ok else "LIVE-ITERATOR CHECKS FAILED"))
sys.exit(0 if ok else 1)
