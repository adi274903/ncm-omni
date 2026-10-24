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
def _s(path, split="train", **kw):
    """Streaming loader, resolved lazily so one bad dataset id cannot kill the producer.

    `path` (not `name`) so callers can pass the config through the `name=` keyword the way
    load_dataset spells it: _s("hf-internal-testing/librispeech_asr_dummy", name="clean").
    """
    def load():
        from datasets import load_dataset
        return load_dataset(path, split=split, streaming=True, **kw)
    return load


def _stream(name, config=None, take=None):
    def load():
        from datasets import load_dataset
        ds = load_dataset(name, config, split="train", streaming=True)
        return ds.take(take) if take else ds
    return load


def _raw(ids, config=None, split="train", column=None, **kw):
    """Like `_stream`, but leaves audio/video **undecoded**.

    `datasets` >= 4 needs torchcodec to decode media columns, and torchcodec only matches a
    narrow range of torch builds. We never want that dependency: the bytes go to soundfile
    (`audio`) or PyAV (`video`) ourselves. Accepts a list of ids and tries them in order, so a
    rename costs one group and not the run.
    """
    if isinstance(ids, str):
        ids = [ids]

    def load():
        from datasets import features, load_dataset
        feat = {"audio": features.Audio(decode=False),
                "video": features.Video(decode=False)}.get(column)
        err = None
        for i in ids:
            try:
                ds = load_dataset(i, config, split=split, streaming=True, **kw)
                return ds.cast_column(column, feat) if feat else ds
            except Exception as e:                             # noqa: BLE001
                err = e
                log(f"[source] {i} ({split}) unavailable: {type(e).__name__}: {str(e)[:70]}")
        raise err
    return load


def _txt(ex):
    for k in ("text", "content", "raw_content", "document"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return {"text": v}
    return {}


def _two(*keys):
    """Join the first two non-empty string fields: problem+solution, question+answer, prompt+response.

    Math, physics and code corpora all split the source material across paired fields; reading only
    `text` on those datasets yields nothing at all.
    """
    def fmt(ex):
        parts = []
        for k in keys:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list) and v and isinstance(v[0], str):
                parts.append(v[0].strip())
        return {"text": "\n\n".join(parts)} if parts else {}
    return fmt


def _img_text(img_keys=("image", "img"), cap_keys=("text", "caption", "captions", "alt_text",
                                                   "question", "label")):
    """Image + the best available string beside it. Field names differ across every vision set."""
    def fmt(ex):
        img = next((ex.get(k) for k in img_keys if ex.get(k) is not None), None)
        if img is None:
            return {}
        for k in cap_keys:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                return {"image": img, "text": v.strip()}
            if isinstance(v, dict):
                inner = next((str(x) for x in v.values() if isinstance(x, str) and x.strip()), "")
                if inner:
                    return {"image": img, "text": inner}
        return {"image": img}
    return fmt


def _img_cap(ex):
    img = ex.get("image")
    if img is None:
        img = ex.get("img")                                    # cifar-style column name
    if img is None:
        return {}
    cap = ""
    for k in ("text", "caption", "captions", "alt_text", "label"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            cap = v
            break
    return {"image": img, "text": str(cap)} if cap else {"image": img}


def _decode_audio(a):
    """(raw bytes|path|already decoded) -> (float32 mono, sr). None means drop the sample."""
    if isinstance(a, dict):
        raw, path = a.get("bytes"), a.get("path")
        try:
            import io
            import soundfile as sf
            wav, sr = sf.read(io.BytesIO(raw) if raw else path, dtype="float32", always_2d=True)
            return wav[:, 0], int(sr)
        except Exception as e:                                 # noqa: BLE001
            log(f"[audio] decode failed: {type(e).__name__}: {str(e)[:70]}")
            return None
    if isinstance(a, (list, tuple)) and len(a) == 2:
        return np.asarray(a[0], dtype=np.float32).flatten(), int(a[1])
    return None


def _audio_text(ex):
    wav = _decode_audio(ex.get("audio"))
    if wav is None:
        return {}
    t = ""
    for k in ("text", "transcription", "transcript", "sentence", "text_normalized",
              "text_original", "raw_transcription"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            t = v
            break
    return {"audio": wav, "text": t} if t else {"audio": wav}


def _local_path(src):
    """hf://datasets/<repo>@<rev>/<file> -> a real local file PyAV can open."""
    if not (isinstance(src, str) and src.startswith("hf://")):
        return src
    from huggingface_hub import hf_hub_download
    body = src[len("hf://datasets/"):]
    repo, rest = body.split("@", 1)
    rev, fname = rest.split("/", 1)
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", revision=rev)


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
            src = v.get("bytes") if isinstance(v, dict) else v
            if not src:
                src = v.get("path") if isinstance(v, dict) else v
            if isinstance(src, (bytes, bytearray)):
                src = io.BytesIO(src)
            src = _local_path(src)
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


# name -> (loader, extractor). Anything that fails to open is skipped with a log line, so a stale
# dataset id degrades the mixture instead of killing the run.
#
# Every id below was opened against the live Hub and its first record decoded (July..Oct 2026
# Hub state). Pruned because they no longer stream at all: lambdalabs/pokemon-blip-captions
# (gated), nlphuji/flickr30k, google/docci, facebook/textvqa, poloclub/diffusiondb, allenai/peS2o
# (all script-based), bigcode/the-stack-smol (gated), huggingface/food101 (renamed ethz/food101),
# HuggingFaceTB/smollm-corpus/python-edu (ships metadata only, no text column).
SOURCES = {
    # --- general + educational text
    "text_fineweb":     (_stream("HuggingFaceFW/fineweb-edu", "sample-10BT"), _txt),
    "text_cosmopedia":  (_stream("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"), _txt),
    "text_wikipedia":   (_stream("wikimedia/wikipedia", "20231101.en"), _txt),
    # --- math
    "text_math_web":    (_stream("open-web-math/open-web-math"), _txt),
    "text_math_fine":   (_stream("HuggingFaceTB/finemath", "finemath-3plus"), _txt),
    "text_math_numina": (_stream("AI-MO/NuminaMath-CoT"), _two("problem", "solution")),
    # --- physics / science papers
    "text_arxiv":       (_stream("ccdv/arxiv-summarization"), _two("article", "abstract")),
    # --- code
    "text_code":        (_stream("codeparrot/codeparrot-clean"), _two("content")),
    "text_code_inst":   (_stream("ise-uiuc/Magicoder-OSS-Instruct-75K"),
                         _two("problem", "solution")),
    # --- vision
    "image_docvqa":     (_s("HuggingFaceM4/DocumentVQA"), _img_text()),
    "image_imagenet":   (_s("evanarlian/imagenet_1k_resized_256"), _img_text()),
    "image_coco":       (_s("detection-datasets/coco"), _img_text()),
    "image_wikiart":    (_s("huggan/wikiart"), _img_text(cap_keys=("artist", "style", "genre"))),
    "image_flowers":    (_s("nelorth/oxford-flowers"), _img_text()),
    "image_food":       (_s("ethz/food101"), _img_text()),
    # --- audio (always decode=False + soundfile)
    "audio_libri":      (_raw("openslr/librispeech_asr", "clean", "train.100",
                              column="audio"), _audio_text),
    "audio_libritts":   (_raw("mythicinfinity/libritts_r", "clean", "train.clean.100",
                              column="audio"), _audio_text),
    "audio_fleurs":     (_raw("google/fleurs", "en_us", column="audio"), _audio_text),
    # "clean", not "microset": the 30x audio budget needs ~31k hours, microset is a demo slice.
    "audio_peoples":    (_raw("MLCommons/peoples_speech", "clean", column="audio"),
                         _audio_text),
    # --- video (data path only; kinetics-mini streams, ucf101-subset is 2 clips)
    "video_ucf":        (_raw(["nateraw/kinetics-mini", "sayakpaul/ucf101-subset"],
                              column="video"), _video_frames),
}
GROUPS = {k: "text" for k in ("text_fineweb", "text_cosmopedia", "text_wikipedia",
                              "text_math_web", "text_math_fine", "text_math_numina",
                              "text_arxiv", "text_code", "text_code_inst")}
GROUPS.update({k: "image" for k in ("image_docvqa", "image_imagenet", "image_coco",
                                    "image_wikiart", "image_flowers", "image_food")})
GROUPS.update({k: "audio" for k in ("audio_libri", "audio_libritts", "audio_fleurs",
                                    "audio_peoples")})
GROUPS["video_ucf"] = "video"


class Source:
    """One stream with retry and a reopen that fast-forwards past what was already read."""

    def __init__(self, name, loader, fmt, shard=(0, 1)):
        self.name, self.loader, self.fmt = name, loader, fmt
        self.n, self.fails, self.dead, self.bad = 0, 0, False, 0
        self.shard, self.sharded = shard, False
        self.it = self._open()

    def _open(self):
        """Split the stream at the *source* level, not after downloading it.

        `ds.shard(N, i)` makes the backend read only its share of the parquet files. Filtering on
        `index % N` instead - which is what the first version did - still downloads every row in
        every worker, so N workers cost N times the bandwidth and run N times slower. The fallback
        filter stays for backends that refuse to shard, but it must not be applied on top of a real
        shard or each worker would see 1/N^2 of the data.
        """
        ds = self.loader()
        if self.shard[1] > 1:
            try:
                ds = ds.shard(num_shards=self.shard[1], index=self.shard[0], contiguous=False)
                self.sharded = True
            except Exception as e:                             # noqa: BLE001
                # Not a defect: `batch()` row-strides unsharded sources by `self.n % N`, so the rows
                # stay correct. The cost is bandwidth - every worker reads the whole stream - which is
                # exactly what the docstring above warns about. Striding again here would compound it
                # into 1/N^2 and silently starve the source.
                log(f"[source] {self.name}: backend shard unavailable "
                    f"({type(e).__name__}: {str(e)[:60]}) - full stream per worker, "
                    f"row stride in batch() (N x bandwidth, correct rows)")
        return iter(ds)

    def reopen(self, err):
        while self.fails < 5:
            self.fails += 1
            log(f"[source] {self.name}: {type(err).__name__}: {str(err)[:90]} -> reopen {self.fails}/5")
            time.sleep(min(30, 4 * self.fails))
            try:
                it = self._open()
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
            self.n += 1                                       # raw index: what reopen() replays
            if not self.sharded and self.shard[1] > 1 and (self.n - 1) % self.shard[1] != self.shard[0]:
                continue
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
    worker, nworkers = cfg.get("worker", (0, 1))
    assert 0 <= worker < nworkers, f"worker {worker} of {nworkers}"
    # Only worker 0 writes the held-out blocks. Every worker reads a disjoint slice, so the rows it
    # holds out are invisible to the others - one copy of each eval file, no duplicates.
    holdout = {g: (n if worker == 0 else 0) for g, n in cfg["holdout"].items()}
    skip = cfg.get("skip_samples", 0)
    mix = {k: float(v) for k, v in cfg["mix"].items() if k in SOURCES}
    only = cfg.get("only_sources")
    if only:
        mix = {k: v for k, v in mix.items() if k in only}
    assert mix, "no usable sources in the mix"

    live, skipped = {}, {}
    for name in mix:
        for attempt in range(4):
            try:
                live[name] = Source(name, SOURCES[name][0], SOURCES[name][1],
                                    shard=(worker, nworkers))
                break
            except Exception as e:                             # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:90]}"
                if attempt == 3:
                    skipped[name] = err
                    log(f"[source] {name} SKIPPED after 4 attempts: {err}")
                else:
                    time.sleep(3 * (attempt + 1))              # 429 from a burst of stream opens
    if not live:
        log("FATAL: no source could be opened")
        return 2
    if skipped:
        log(f"WARNING: {len(skipped)} source(s) missing - the mix is NOT the configured one:",
            list(skipped))
    tot = sum(mix[n] for n in live)
    frac = {n: mix[n] / tot for n in live}
    log("sources:", {n: round(frac[n], 3) for n in live})
    needed = {GROUPS[n] for n in live}        # only groups with a live source gate EVAL_READY

    # Per-worker seed. One shared seed made every worker walk the sources in the same order with
    # the same weights, so N workers piled onto the same heavy source at the same instant - the
    # thundering herd that makes a 4-worker run hungrier and no faster than a 1-worker run.
    rng = random.Random(int(cfg["seed"]) + 7919 * worker)
    used = {n: 0 for n in live}
    dropped = {n: 0 for n in live}
    held = {g: [] for g in holdout}
    shuf, cur, shard_idx, written = [], [], 0, 0
    seen = 0
    t0 = time.time()

    def next_shard_path():
        nonlocal shard_idx
        tag = f"_w{worker}" if nworkers > 1 else ""
        p = os.path.join(out, f"shard{tag}_{shard_idx:06d}.npz")
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
                mine = f"shard_w{worker}_" if nworkers > 1 else "shard_"
                while (sum(1 for f in os.listdir(out)
                           if f.startswith(mine) and f.endswith(".npz")) >= max_ahead):
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
                    if nworkers == 1:
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
    st = dict(pull=0.0, enc=0.0, write=0.0)        # wall clock per stage, flushed into progress

    while active and seen_so_far[0] < budget:
        name = rng.choices(active, weights=[frac[n] for n in active])[0]
        group = GROUPS[name]
        _t = time.perf_counter()
        batch = live[name].batch(cfg["batch_docs"])
        st["pull"] += time.perf_counter() - _t
        if not batch:
            active.remove(name)
            log(f"[source] {name} exhausted/dead")
            continue
        for parts in batch:
            if seen_so_far[0] >= budget:
                break
            used[name] += 1
            _t = time.perf_counter()
            codes, present = encode_sample(codecs, sched, parts)
            st["enc"] += time.perf_counter() - _t
            if not any(present.values()):
                dropped[name] += 1
                continue
            emit({"codes": codes, "present": present}, group)
        _t = time.perf_counter()
        flush()
        st["write"] += time.perf_counter() - _t
        el = max(1e-9, time.time() - t0)
        tot = max(1e-9, sum(st.values()))
        json.dump({"samples": seen_so_far[0], "budget": budget, "shards": written,
                   "per_source": used,
                   # rows/s per source: a source that is alive but starving shows up here without
                   # anyone having to diff two progress files by hand.
                   "per_source_s": {n: round(v / el, 3) for n, v in used.items() if v},
                   "dropped": {n: v for n, v in dropped.items() if v},
                   "eval_held": {g: len(v) for g, v in held.items()},
                   "elapsed_min": round(el / 60, 2),
                   "samples_per_s": round(seen_so_far[0] / el, 3),
                   # Where the wall clock went: pull = blocked on the Hub, enc = decode + codec,
                   # write = npz. A produce run that looks stuck is almost always pull.
                   "stage_pct": {k: round(100 * v / tot, 1) for k, v in st.items()},
                   "stage_s": {k: round(v, 1) for k, v in st.items()}},
                  open(os.path.join(out, "progress.json" if nworkers == 1
                                    else f"progress_w{worker}.json"), "w"))
        if seen_so_far[0] % 25 == 0:
            raw = {n: live[n].n for n in live}
            log(f"progress {seen_so_far[0]}/{budget} samples | {written} shards | "
                f"{seen_so_far[0] / el:.2f} samples/s | "
                f"pull {100 * st['pull'] / tot:.0f}% enc {100 * st['enc'] / tot:.0f}% "
                f"write {100 * st['write'] / tot:.0f}% | dropped {sum(dropped.values())} | "
                f"pulled {sum(raw.values())} rows | sharded="
                f"{sum(1 for n in live if live[n].sharded)}/{len(live)}")

    for row in shuf:                                           # flush the shuffle buffer
        cur.append(row)
    flush(final=True)
    for g, rows in held.items():                               # under-filled groups still get a file
        if rows and len(rows) < holdout.get(g, 0):
            atomic_npz(os.path.join(out, f"eval_{g}.npz"), stack_samples(rows, sched), meta)
    stats = dict(samples=seen_so_far[0], shards=written, per_source=used, dropped=dropped,
                 budget=budget, skip_samples=skip, schedule=meta, worker=[worker, nworkers],
                 skipped_sources=skipped,
                 elapsed_min=round((time.time() - t0) / 60, 2))
    if nworkers == 1:                                          # single process: owns the sentinels
        open(os.path.join(out, "EVAL_READY"), "w").write("1")
        json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
        open(os.path.join(out, "DONE"), "w").write("1")
    else:                                                       # coordinator aggregates and decides
        json.dump(stats, open(os.path.join(out, f"stats_w{worker}.json"), "w"), indent=2)
    log("DONE", {k: stats[k] for k in ("samples", "shards", "elapsed_min")})
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--worker", default=None, help="i/N - this process's slice of every stream")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    if a.worker:
        i, n = a.worker.split("/")
        cfg["worker"] = (int(i), int(n))
    try:
        return run(cfg, selfcheck=a.selfcheck)
    except SystemExit:
        raise
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
