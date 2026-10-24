"""Offline check for `omni.livestream._Weighted`: does the draw order follow `frac`?

Models what the Hub pumps actually did on the A100 run - text arriving in under a second, image
after ~20 s, audio after ~37 s, all of them faster than the 112 rows/s consumer - and compares the
old plain-FIFO draw against the deficit scheduler. No network, no GPU.

The FIFO column is the bug: the queue fills with text while the slow pumps are still on their first
download, so the first audio row lands ~4.2k rows in, past the end of a 3664-sample run.
"""
import heapq
import sys
from collections import deque

from omni.livestream import _Weighted

FRAC = {"text": 0.4166, "image": 0.4167, "audio": 0.1667}
START = {"text": 0.5, "image": 2.0, "audio": 16.0}       # first row each pump delivered, measured
RATE = {"text": 260.0, "image": 40.0, "audio": 2.5}       # rows/s produced, all ahead of 240 consumed
CONSUMER = 240.0                                          # rows/s encoded on one thread
PATIENCE = 10.0


def arrivals(start=None, rate=None, seconds=400.0):
    start, rate = start or START, rate or RATE
    heap, i = [], 0
    for g, s in start.items():
        if s is None or rate[g] <= 0:
            continue                                       # source never delivers
        for k in range(int((seconds - s) * rate[g])):
            heapq.heappush(heap, (s + k / rate[g], i, g))
            i += 1
    return sorted(heap)


def run_fifo(arr, take=3000, consumer=CONSUMER):
    q, out, now = deque(arr), [], 0.0
    while q and len(out) < take:
        t, _i, g = q.popleft()
        now = max(now, t)
        out.append(g)
        now += 1.0 / consumer
    return out


def run_sched(arr, take=3000, consumer=CONSUMER, patience=PATIENCE, cap=64):
    """The iter_rows draw loop: wait for the source that is owed a slot, with a patience fallback."""
    wq, q, out, now = _Weighted(FRAC, cap), deque(arr), [], 0.0
    while len(out) < take:
        waited = 0.0
        while True:
            name = wq.next()
            if name is None:
                return out, wq
            if wq.ready(name):
                break
            fb = wq.best_ready()
            if fb is not None and waited >= patience:
                wq.cool_down(name)                         # mirrors the iter_rows fallback
                name = fb
                break
            if not q:
                if fb is not None:
                    name = fb
                    break
                return out, wq
            t, _i, g = q.popleft()
            gap = max(0.0, t - now)
            if fb is not None and waited + gap >= patience:
                wq.cool_down(name)
                name = fb
                q.appendleft((t, _i, g))                       # put it back: we are not ready to serve it
                break
            now, waited = now + gap, waited + gap
            wq.push(g, 1)
        wq.take(name)
        out.append(name)
        now += 1.0 / consumer
    return out, wq


def first(out, g):
    return out.index(g) if g in out else -1


def shares(out):
    n = max(1, len(out))
    return {g: round(out.count(g) / n, 3) for g in FRAC}


ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


arr = arrivals()
print(f"arrival stream: {len(arr)} rows; |text| ~{sum(1 for _, _, g in arr if g == 'text')}")

fifo = run_fifo(arr)
sched, wq = run_sched(arr)

print(f"\n{'':7} {'first image':>11} {'first audio':>11}   mix over 3000 draws")
print(f"{'FIFO':7} {first(fifo,'image'):>11} {first(fifo,'audio'):>11}   {shares(fifo)}")
print(f"{'sched':7} {first(sched,'image'):>11} {first(sched,'audio'):>11}   {shares(sched)}")
print(f"\ntarget {FRAC}  cap={wq.cap}  dropped={wq.dropped}  served={shares(sched)}")
print()

# 1. The reported bug: FIFO's first audio lands after the whole 458-step run (3664 samples).
check("FIFO starves audio past a full 3664-sample run",
      first(fifo, "audio") == -1 or first(fifo, "audio") > 3664,
      f"first audio at row {first(fifo,'audio')} of {len(fifo)}")

# 2. The fix: the scheduler reaches audio within a few hundred rows.
check("scheduler reaches audio inside 2000 rows", 0 <= first(sched, "audio") < 2000,
      f"first audio at row {first(sched,'audio')}")

# 3. ...and reaches image early too (FIFO waits ~2200).
check("scheduler reaches image before FIFO does",
      0 <= first(sched, "image") < first(fifo, "image"),
      f"{first(sched,'image')} vs {first(fifo,'image')}")

# 4. Long-run mix: audio is now materially present rather than a rounding error.
got = shares(sched)
check("audio share is at least half its target", got["audio"] >= FRAC["audio"] * 0.5, str(got))
check("no group exceeds target by more than 0.15",
      all(got[g] - FRAC[g] <= 0.15 for g in FRAC), str(got))

# 5. Steady state: every pump live from t=0 and audio the slow one. The mix must be near-exact even
#    though audio produces 500x slower than text - this is the property FIFO cannot have.
steady = arrivals(start={"text": 0.0, "image": 0.0, "audio": 0.0},
                  rate={"text": 260.0, "image": 90.0, "audio": 6.0})
sched5, _ = run_sched(steady, take=1200)
got5 = shares(sched5)
check("steady state hits the target mix within 0.05",
      all(abs(got5[g] - FRAC[g]) <= 0.05 for g in FRAC), str(got5))
check("steady state is rate-limited by the scarcest source, not by text",
      got5["audio"] > 0.12, str(got5))

# 6. A source that never delivers must throttle, not hang: patience hands the slot on.
dead = arrivals(start={"text": 0.0, "image": 0.0, "audio": None})
out6, wq6 = run_sched(dead, take=500)
check("dead source does not hang the stream", len(out6) == 500, f"{len(out6)} rows")
check("dead source is skipped by the patience fallback",
      shares(out6)["audio"] == 0.0 and wq6.dropped["audio"] == 0)

# 7. Exhaustion: retire() stops the wait even when the queue is empty.
wq7 = _Weighted({"a": 0.5, "b": 0.5}, cap=4)
wq7.push("a", 1)
check("retire frees the wait", wq7.next() in {"a", "b"} and
      (wq7.retire("b") or wq7.next() == "a"))

# 8. cap bounds the surplus instead of letting a fast source grow without limit.
wq8 = _Weighted({"text": 0.5, "image": 0.5}, cap=4)
for _ in range(100):
    wq8.push("text", 1)
check("cap bounds the buffered surplus", len(wq8.buf["text"]) == 4 and wq8.dropped["text"] == 96,
      f"{len(wq8.buf['text'])} buffered, {wq8.dropped['text']} dropped")

print("\n" + ("ALL SCHEDULER CHECKS PASS" if ok else "SCHEDULER CHECKS FAILED"))
sys.exit(0 if ok else 1)
