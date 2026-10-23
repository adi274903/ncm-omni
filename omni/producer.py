"""Omni streaming producer: HF sources -> frozen codecs -> nested code shards.

Same architecture as the DAPT stream producer, with one structural change: the unit written to
disk is not a 2048-token text block but a SAMPLE carrying K code levels for each modality that
sample contains (text / image / audio / video). Shards are .npz, self-describing (the schedule
travels in every file), and the training process can start as soon as EVAL_READY appears.

Why a separate process: image pyramid + mel encoding are Python/numpy heavy and would fight the
training loop for the GIL. Why rolling shards and not one big precompute: you can start training
a minute in, and a crash costs one shard, not a CPU-day.

Run:  python -m omni.producer cfg.json [--selfcheck]
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np

from omni.codecs import build_codec
from omni.schedule import MODALITIES, OmniSchedule

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ sources
def _s(path, **kw):
    """Streaming loader, resolved lazily so one bad dataset id cannot kill the producer.

    `path` (not `name`) so callers can pass the config through the `name=` keyword the way
    load_dataset spells it: _s("hf-internal-testing/librispeech_asr_dummy", name="clean").
    """
    def load():
        from datasets import load_dataset
        return load_dataset(path, split="train", streaming=True, **kw)
    return load


def _stream(name, config=None, take=None):
    def load():
        from datasets import load_dataset
        ds = load_dataset(name, config, split="train", streaming=True)
        return ds.take(take) if take else ds
    return load


def _first(ids, config=None):
    """Try dataset ids in order; the first that opens wins. Stale ids degrade, never kill."""
    def load():
        from datasets import load_dataset
        err = None
        for i in ids:
            try:
                return load_dataset(i, config, split="train", streaming=True)
            except Exception as e:                             # noqa: BLE001
                err = e
                log(f"[source] id {i} unavailable: {type(e).__name__}: {str(e)[:70]}")
        raise err
    return load


def _txt(ex):
    for k in ("text", "content", "raw_content", "document"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return {"text": v}
    return {}


def _img_cap(ex):
    img = ex.get("image")
    if img is None:
        return {}
    cap = ""
    for k in ("text", "caption", "captions", "alt_text"):
        v = ex.get(k)
        if isinstance(v, str):
            cap = v
            break
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            cap = v[0]
            break
    return {"image": img, "text": cap} if cap else {"image": img}


def _audio_text(ex):
    a = ex.get("audio")
    if a is None:
        return {}
    t = ""
    for k in ("text", "transcription", "transcript", "sentence"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            t = v
            break
    return {"audio": a, "text": t} if t else {"audio": a}


def _video_frames(ex, max_frames=8):
    """Clip -> list of PIL frames. Video is data-path only, so a decode failure just drops it."""
    v = ex.get("video") or ex.get("videos") or ex.get("frames")
    if v is None:
        return {}
    if isinstance(v, (list, tuple)) and v and hasattr(v[0], "convert"):
        frames = list(v)[:max_frames]                         # already decoded by `datasets`
    else:
        try:
            import io
            import av
            src = v.get("bytes") if isinstance(v, dict) else None
            if not src:
                src = v.get("path") if isinstance(v, dict) else v
            if isinstance(src, (bytes, bytearray)):
                src = io.BytesIO(src)
            if src is None:
                return {}
            frames = []
            with av.open(src) as c:
                for i, fr in enumerate(c.decode(video=0)):
                    if i % 3 == 0:
                        frames.append(fr.to_image())
                    if len(frames) >= max_frames:
                        break
        except Exception as e:                                # noqa: BLE001
            log(f"[video] decode failed: {type(e).__name__}: {str(e)[:70]}")
            return {}
    if len(frames) < 2:
        return {}
    t = ex.get("text") or ex.get("caption")
    return {"video": frames, "text": t} if isinstance(t, str) and t.strip() else {"video": frames}


# name -> (group, loader, extractor). Anything that fails to open is skipped with a log line,
# so a stale dataset id degrades the mixture instead of killing the run.
SOURCES = {
    "text_fineweb":  (_stream("HuggingFaceFW/fineweb-edu", "sample-10BT"), _txt),
    "text_wikitext": (_stream("Salesforce/wikitext", "wikitext-103-raw-v1"), _txt),
    "image_pokemon": (_s("lambdalabs/pokemon-blip-captions"), _img_cap),
    "image_flickr":  (_s("nlphuji/flickr30k"), _img_cap),
    "audio_dummy":   (_s("hf-internal-testing/librispeech_asr_dummy", name="clean"), _audio_text),
    "audio_libri":   (_stream("openslr/librispeech_asr", "clean"), _audio_text),
    # video ids are the least stable thing in this file: `_first` tries each and logs what failed,
    # so a rename costs the video group and nothing else.
    "video_ucf":     (_first(["sayakpaul/ucf101-subset",
                              "HuggingFaceM4/webvid",
                              "hf-internal-testing/small-video-dataset"]), _video_frames),
}
GROUPS = {"text_fineweb": "text", "text_wikitext": "text",
          "image_pokemon": "image", "image_flickr": "image",
          "audio_dummy": "audio", "audio_libri": "audio",
          "video_ucf": "video"}


class Source:
    """One stream with retry and a reopen that fast-forwards past what was already read."""

    def __init__(self, name, loader, fmt):
        self.name, self.loader, self.fmt = name, loader, fmt
        self.n, self.fails, self.dead, self.bad = 0, 0, False, 0
        self.it = iter(loader())

    def reopen(self, err):
        while self.fails < 5:
            self.fails += 1
            log(f"[source] {self.name}: {type(err).__name__}: {str(err)[:90]} -> reopen {self.fails}/5")
            time.sleep(min(30, 4 * self.fails))
            try:
                it = iter(self.loader())
                for _ in range(self.n):                        # fast-forward, no encoding
                    next(it)
                self.it = it
                return True
            except Exception as e:                             # noqa: BLE001
                err = e
        return False

    def batch(self, k):
        out = []
        while len(out) < k and not self.dead:
            try:
                ex = next(self.it)
            except StopIteration:
                self.dead = True
                break
            except Exception as e:                             # noqa: BLE001
                if not self.reopen(e):
                    self.dead = True
                    break
                continue
            self.n += 1
            try:
                d = self.fmt(ex)
            except Exception:                                  # noqa: BLE001
                self.bad += 1
                continue
            if d:
                out.append(d)
        return out


# ------------------------------------------------------------------ encoding
def make_codecs(sched, V, prefer_real=True):
    return {m: build_codec(m, sched.K, sched.R, V, sched.base_len[m], prefer_real)
            for m in MODALITIES}


def encode_sample(codecs, sched, parts):
    """dict modality -> raw object  =>  dict modality -> int32[flat_len], plus presence vector."""
    codes, present = {}, {}
    for m in MODALITIES:
        obj = parts.get(m)
        ok = obj is not None
        if ok:
            try:
                lv = codecs[m].encode(obj)
            except Exception as e:                             # noqa: BLE001
                log(f"[encode] {m}: {type(e).__name__}: {str(e)[:80]}")
                ok = False
        if ok:
            flat = np.concatenate([np.asarray(x, dtype=np.int32).ravel() for x in lv])
            if flat.size != sched.flat_len(m):                 # codec disagreeing with schedule
                log(f"[encode] {m}: got {flat.size}, schedule wants {sched.flat_len(m)}")
                ok = False
            else:
                codes[m] = np.clip(flat, 1, codecs[m].V)           # 0 is reserved for NULL
        if not ok:
            codes[m] = np.full(sched.flat_len(m), sched.null_code, dtype=np.int32)
        present[m] = bool(ok)
    return codes, present


def stack_samples(rows, sched, mods=MODALITIES):
    out = {}
    for m in mods:
        out[m] = np.stack([r["codes"][m] for r in rows]).astype(np.int32)
        out[f"{m}_present"] = np.array([r["present"][m] for r in rows], dtype=bool)
    return out


def schedule_meta(sched, codecs, V):
    return dict(K=sched.K, R=sched.R, null_code=sched.null_code,
                base_len=dict(sched.base_len), modalities=list(mods_default()),
                level_lens={m: list(sched.level_lens(m)) for m in MODALITIES},
                offsets={m: list(sched.offsets(m)) for m in MODALITIES},
                vocab={m: codecs[m].V for m in MODALITIES},
                codecs={m: codecs[m].info() for m in MODALITIES},
                version=1)


def mods_default():
    return MODALITIES


def atomic_npz(path, arrays, meta):
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ main
def run(cfg, selfcheck=False):
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    sched = OmniSchedule(base_len=cfg["base_len"], K=cfg["K"], R=cfg["R"],
                         null_code=cfg.get("null_code", 0))
    V = cfg["codebook_size"]
    codecs = make_codecs(sched, V, cfg.get("prefer_real", True))
    meta = schedule_meta(sched, codecs, V)
    log(sched.describe())
    log("codecs:", {m: codecs[m].name for m in MODALITIES})

    if selfcheck:
        # Every modality, with synthetic objects only: no network, no dataset ids. What is being
        # proved here is the ladder contract - len(level_k) == schedule.level_len(m, k) - and
        # that no code ever collides with the NULL slot.
        import numpy as _np
        from PIL import Image as _Image

        def _img(seed):
            g = _np.random.default_rng(seed)
            a = (g.random((32, 32, 3)) * 255).astype("uint8")
            return _Image.fromarray(a)

        parts = [
            {"text": "a blood pressure of 140 over 90 is stage two hypertension",
             "image": _img(0),
             "video": [_img(i) for i in range(8)],
             "audio": (_np.sin(_np.linspace(0, 400, 16000)).astype("float32"), 16000)},
            {"text": "short", "image": _img(1)},
        ]
        rows = [dict(zip(("codes", "present"), encode_sample(codecs, sched, p))) for p in parts]
        for m in MODALITIES:
            assert rows[0]["present"][m], f"{m}: synthetic sample failed to encode"
            assert len(rows[0]["codes"][m]) == sched.flat_len(m), \
                f"{m}: flat {len(rows[0]['codes'][m])} != schedule {sched.flat_len(m)}"
            c = rows[0]["codes"][m]
            assert int(c.min()) >= 1, f"{m}: a code hit the NULL slot"
            assert int(c.max()) <= codecs[m].V, f"{m}: code {int(c.max())} > codebook {codecs[m].V}"
            off = 0
            for k in range(sched.K):
                l = sched.level_len(m, k)
                assert len(rows[0]["codes"][m][off:off + l]) == l, f"{m} level {k} length"
                off += l
        assert not rows[1]["present"]["audio"], "audio should be absent for a text-only sample"

        arr = stack_samples(rows, sched)
        atomic_npz(os.path.join(out, "selfcheck.npz"), arr, meta)
        log("selfcheck wrote", os.path.join(out, "selfcheck.npz"),
            {k: v.shape for k, v in arr.items()})
        log("selfcheck: ladder contract holds for", {m: codecs[m].name for m in MODALITIES})
        return 0

    budget = cfg["sample_budget"]
    shard_n = cfg["shard_samples"]
    max_ahead = cfg["max_ahead_shards"]
    holdout = cfg["holdout"]
    skip = cfg.get("skip_samples", 0)
    mix = {k: float(v) for k, v in cfg["mix"].items() if k in SOURCES}
    only = cfg.get("only_sources")
    if only:
        mix = {k: v for k, v in mix.items() if k in only}
    assert mix, "no usable sources in the mix"

    live = {}
    for name in mix:
        try:
            live[name] = Source(name, SOURCES[name][0], SOURCES[name][1])
        except Exception as e:                                 # noqa: BLE001
            log(f"[source] {name} skipped: {type(e).__name__}: {str(e)[:100]}")
    if not live:
        log("FATAL: no source could be opened")
        return 2
    tot = sum(mix[n] for n in live)
    frac = {n: mix[n] / tot for n in live}
    log("sources:", {n: round(frac[n], 3) for n in live})
    needed = {GROUPS[n] for n in live}        # only groups with a live source gate EVAL_READY

    rng = random.Random(cfg["seed"])
    used = {n: 0 for n in live}
    dropped = {n: 0 for n in live}
    held = {g: [] for g in holdout}
    shuf, cur, shard_idx, written = [], [], 0, 0
    seen = 0
    t0 = time.time()

    def next_shard_path():
        nonlocal shard_idx
        p = os.path.join(out, f"shard_{shard_idx:06d}.npz")
        shard_idx += 1
        return p

    def flush(final=False):
        nonlocal cur, written
        while len(cur) >= shard_n or (final and cur):
            rows, cur = cur[:shard_n], cur[shard_n:]
            if not rows:
                break
            arr = stack_samples(rows, sched)
            if seen_so_far[0] > skip:                          # never rewrite what a resume already trained on
                while (sum(1 for f in os.listdir(out)
                           if f.startswith("shard_") and f.endswith(".npz")) >= max_ahead):
                    if os.path.exists(os.path.join(out, "STOP")):
                        log("STOP seen, exiting")
                        sys.exit(0)
                    time.sleep(1)
                atomic_npz(next_shard_path(), arr, meta)
                written += 1

    seen_so_far = [0]

    def emit(row, group):
        """Hold out the first N samples per group, then shuffle-buffer the rest into shards."""
        if len(held[group]) < holdout.get(group, 0):
            held[group].append(row)
            if len(held[group]) == holdout[group]:
                atomic_npz(os.path.join(out, f"eval_{group}.npz"),
                           stack_samples(held[group], sched), meta)
                log(f"[eval] {group}: {holdout[group]} held out")
                if all(len(held.get(g, [])) >= holdout.get(g, 0) for g in needed):
                    open(os.path.join(out, "EVAL_READY"), "w").write("1")
            return
        seen_so_far[0] += 1
        if len(shuf) < cfg["shuffle_buf"]:
            shuf.append(row)
            return
        j = rng.randrange(len(shuf))
        out_row, shuf[j] = shuf[j], row
        cur.append(out_row)

    active = list(live)
    while active and seen_so_far[0] < budget:
        name = rng.choices(active, weights=[frac[n] for n in active])[0]
        group = GROUPS[name]
        batch = live[name].batch(cfg["batch_docs"])
        if not batch:
            active.remove(name)
            log(f"[source] {name} exhausted/dead")
            continue
        for parts in batch:
            if seen_so_far[0] >= budget:
                break
            used[name] += 1
            codes, present = encode_sample(codecs, sched, parts)
            if not any(present.values()):
                dropped[name] += 1
                continue
            emit({"codes": codes, "present": present}, group)
        flush()
        json.dump({"samples": seen_so_far[0], "budget": budget, "shards": written,
                   "per_source": used, "eval_held": {g: len(v) for g, v in held.items()},
                   "elapsed_min": round((time.time() - t0) / 60, 2),
                   "samples_per_s": round(seen_so_far[0] / max(1e-9, time.time() - t0), 1)},
                  open(os.path.join(out, "progress.json"), "w"))
        if seen_so_far[0] % 100 == 0:
            log(f"progress {seen_so_far[0]}/{budget} samples | {written} shards | "
                f"{seen_so_far[0] / max(1e-9, time.time() - t0):.1f} samples/s")

    for row in shuf:                                           # flush the shuffle buffer
        cur.append(row)
    flush(final=True)
    for g, rows in held.items():                               # under-filled groups still get a file
        if rows and len(rows) < holdout.get(g, 0):
            atomic_npz(os.path.join(out, f"eval_{g}.npz"), stack_samples(rows, sched), meta)
    open(os.path.join(out, "EVAL_READY"), "w").write("1")
    stats = dict(samples=seen_so_far[0], shards=written, per_source=used, dropped=dropped,
                 budget=budget, skip_samples=skip, schedule=meta,
                 elapsed_min=round((time.time() - t0) / 60, 2))
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    open(os.path.join(out, "DONE"), "w").write("1")
    log("DONE", {k: stats[k] for k in ("samples", "shards", "elapsed_min")})
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    try:
        return run(cfg, selfcheck=a.selfcheck)
    except SystemExit:
        raise
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
