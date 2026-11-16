# %% [markdown]
# ## 4. Data: streamed from the Hub
#
# ```
# Hub (streaming=True) -> W worker processes (decode, cheap) -> micro-batch builder
#      -> GPU codecs on a side CUDA stream -> RAM ring (PREFETCH batches) -> trainer
# ```
#
# * **No raw data and no dataset cache are written.** The only buffer is the in-RAM ring; the encoded codes can be
#   recorded as shards (next section).
# * Rows are **modality-homogeneous** (text-only, prompt->response, caption->image, transcript->speech),
#   so there are exactly 4 static pack shapes and no wasted NULL slots.
# * The stream is split 3 ways by shard (or by row index when a dataset has few shards):
#   part 0 = validation, part 1 = codebook fit, parts 2.. = training workers. Splits never overlap.
# * A source that errors is retried with backoff and dropped after repeated failures (gated or renamed
#   datasets cost their own weight, not the run).
#
# Sources are what open pretraining recipes use: FineWeb-Edu / Wikipedia / Cosmopedia / code / OpenWebMath
# (text), NuminaMath + SmolTalk (instruction), CC12M / CC3M / COCO / DocVQA (image+caption),
# LibriSpeech / LibriTTS-R / People's Speech (audio+transcript).

# %%
import re as _re


@dataclass
class Source:
    name: str
    kind: str                      # text0 | text | image | audio
    path: str
    config: str = None
    split: str = "train"
    weight: float = 1.0
    raw: dict = None               # column -> "image" | "audio" (kept as undecoded bytes)
    buf: int = 512                 # shuffle buffer (rows, in RAM)
    fn: str = ""                   # extractor name


# ---- which sources ---------------------------------------------------------------------------
# "lean" (default): ONE corpus per modality, so a codec/ladder change has no source-mix confound and
#   a gain can only come from the thing you changed.
#     text  fineweb-edu sample-10BT       - the strongest open filtered web corpus
#     image ImageNet-1k resized 256        - caption = the class label, i.e. a 1000-way DETERMINISTIC
#                                            function of the image. That is a cleaner caption test than
#                                            CC12M alt-text, which is why image looked like it used
#                                            its caption while using only the caption's presence.
#     audio LibriSpeech 960 h (3 splits)   - clean read speech, sentence-level clips that fit the 8 s
#                                            window. 9.6x the distinct audio of train.100 alone, which
#                                            is the direct response to the T1 audio overfit.
# "rich": the original multi-source mix. NCM_SRC_SET=rich.
SRC_SET = _knob("SRC_SET", "lean")
KEEP_INSTRUCT = _knob("KEEP_INSTRUCT", False)   # fineweb-edu has no prompts: without this there are no
                                                # prompt->response text rows and the text prefix test
                                                # cannot run. Adds HuggingFaceTB/smoltalk (small).
AUDIO_SRC = _knob("AUDIO_SRC", "peoples")   # librispeech | libritts | peoples | gigaspeech | emilia

# GigaSpeech (10k h) and Emilia (101k h) are the modern large corpora, but both are GATED: the Hub
# refuses them (auto-gate) until the terms are accepted on the dataset page. Swap AUDIO_SRC once they are.
AUDIO_SETS = {
    # People's Speech is the largest OPEN transcript-aligned corpus on the Hub (~30k h across clean +
    # dirty). GigaSpeech (10k h) and Emilia (101k h) are gated; espnet/yodas and yodas2 are DEAD
    # ("dataset scripts are no longer supported"); MLS ships no `english` config any more.
    # MEASURED with this dataset's own duration_ms: median 14.8 s, p10 14.3, p90 15.0 - and only 1%
    # of clips fit the old 8.19 s window. That is why the source held weight 0.85 and still left audio
    # overfitting: 99% of it was thrown away at the door. AUDIO_LEN=1024 frames = 16.4 s covers it.
    "peoples": [
        Source("peoples_clean", "audio", "MLCommons/peoples_speech", "clean", weight=0.60,
               raw={"audio": "audio"}, buf=64, fn="a_text"),
        Source("peoples_dirty", "audio", "MLCommons/peoples_speech", "dirty", weight=0.40,
               raw={"audio": "audio"}, buf=64, fn="a_text"),
    ],
    "librispeech": [
        Source("libri100", "audio", "openslr/librispeech_asr", "all", "train.clean.100", weight=1 / 3,
               raw={"audio": "audio"}, buf=64, fn="a_text"),
        Source("libri360", "audio", "openslr/librispeech_asr", "all", "train.clean.360", weight=1 / 3,
               raw={"audio": "audio"}, buf=64, fn="a_text"),
        Source("libri500", "audio", "openslr/librispeech_asr", "all", "train.other.500", weight=1 / 3,
               raw={"audio": "audio"}, buf=64, fn="a_text"),
    ],
    "libritts": [Source("libritts", "audio", "mythicinfinity/libritts_r", "clean", "train.clean.100",
                        weight=1.0, raw={"audio": "audio"}, buf=64, fn="a_tts")],
    "gigaspeech": [Source("giga", "audio", "speechcolab/gigaspeech", "s", "train", weight=1.0,
                          raw={"audio": "audio"}, buf=64, fn="a_text")],
    "emilia": [Source("emilia", "audio", "amphion/Emilia-Dataset", weight=1.0,
                      raw={"audio": "audio"}, buf=64, fn="a_text")],
}
if AUDIO_SRC not in AUDIO_SETS:
    raise SystemExit(f"NCM_AUDIO_SRC={AUDIO_SRC!r} unknown; pick from {sorted(AUDIO_SETS)}")

if SRC_SET == "lean":
    # Largest sources that carry image BYTES. Measured against the Hub: DataComp-1B (1.28B) and
    # Recap-DataComp-1B are `url` + caption only, conceptual_captions is image_url + caption, LAION-2B
    # is gated. So CC12M + CC3M = 15.3M captioned images is the open ceiling that streams as bytes,
    # and it is 12x ImageNet-1k with real captions instead of a class label.
    SOURCES = [
        Source("fineweb", "text0", "HuggingFaceFW/fineweb-edu", "sample-10BT", weight=1.0, fn="t_text"),
        Source("cc12m", "image", "pixparse/cc12m-wds", weight=12 / 15.3, raw={"jpg": "image"},
               buf=128, fn="i_wds"),
        Source("cc3m", "image", "pixparse/cc3m-wds", weight=3.3 / 15.3, raw={"jpg": "image"},
               buf=128, fn="i_wds"),
    ] + AUDIO_SETS[AUDIO_SRC]
    if KEEP_INSTRUCT:
        SOURCES.append(Source("smoltalk", "text", "HuggingFaceTB/smoltalk", "all", weight=1.0,
                              fn="p_smoltalk"))
else:
    SOURCES = [
        # --- text, plain pretraining (text0)
        Source("fineweb", "text0", "HuggingFaceFW/fineweb-edu", "sample-10BT", weight=0.45, fn="t_text"),
        Source("wiki", "text0", "wikimedia/wikipedia", "20231101.en", weight=0.15, fn="t_text"),
        Source("cosmo", "text0", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", weight=0.15, fn="t_text"),
        Source("code", "text0", "codeparrot/codeparrot-clean", weight=0.10, fn="t_content"),
        Source("owm", "text0", "open-web-math/open-web-math", weight=0.15, fn="t_text"),
        # --- text, prompt -> response
        Source("numina", "text", "AI-MO/NuminaMath-CoT", weight=0.5, fn="p_numina"),
        Source("smoltalk", "text", "HuggingFaceTB/smoltalk", "all", weight=0.5, fn="p_smoltalk"),
        # --- image + caption
        # weights follow how much DISTINCT data a source has: a small source at a high weight is repeated tens of times in a short run and gets memorised
        Source("cc12m", "image", "pixparse/cc12m-wds", weight=0.60, raw={"jpg": "image"}, buf=128, fn="i_wds"),
        Source("cc3m", "image", "pixparse/cc3m-wds", weight=0.33, raw={"jpg": "image"}, buf=128, fn="i_wds"),
        Source("docvqa", "image", "HuggingFaceM4/DocumentVQA", weight=0.05, raw={"image": "image"}, buf=64, fn="i_docvqa"),
        Source("coco", "image", "lmms-lab/COCO-Caption2017", split="val", weight=0.02, raw={"image": "image"}, buf=64, fn="i_coco"),
        # --- audio + transcript (clips longer than the 8 s window are dropped, so the Libri sets keep only a fraction of their rows)
        Source("libri", "audio", "openslr/librispeech_asr", "clean", "train.100", weight=0.05, raw={"audio": "audio"}, buf=64, fn="a_text"),
        Source("libritts", "audio", "mythicinfinity/libritts_r", "clean", "train.clean.100", weight=0.10, raw={"audio": "audio"}, buf=64, fn="a_tts"),
        Source("peoples", "audio", "MLCommons/peoples_speech", "clean", weight=0.85, raw={"audio": "audio"}, buf=64, fn="a_text"),
    ]
for _s in SOURCES:
    if _s.name in SOURCE_WEIGHTS:
        _s.weight = float(SOURCE_WEIGHTS[_s.name])
SOURCES = [s for s in SOURCES if s.weight > 0]
_kinds = sorted({s.kind for s in SOURCES})
print(f"source set {SRC_SET}: {len(SOURCES)} sources " + (f"| audio {AUDIO_SRC}" if SRC_SET == "lean" else ""))
for _s in SOURCES:
    print(f"  {_s.kind:6s} {_s.name:10s} {_s.path}"
          + (f" | config {_s.config}" if _s.config else "") + f" | split {_s.split}")
_missing = [r for r in ROW_TYPES if r not in _kinds]
if _missing:
    print(f"  !! no source produces row type(s) {_missing}. Those rows never appear, their share of "
          f"the code budget is redistributed, and any check that needs them cannot run."
          + (" Set NCM_KEEP_INSTRUCT=1 to add a prompt->response text source." if "text" in _missing else ""))


def _bytes(s, n):
    b = (s or "").encode("utf-8", "ignore")[:n]
    a = np.zeros(n, np.uint8)
    a[:len(b)] = np.frombuffer(b, np.uint8)
    return a, len(b)


def _clean(s):
    return _re.sub(r"\s+", " ", s or "").strip()


# ---- extractors: HF example -> list of row dicts (numpy / bytes only, so they pickle cheaply) ----
TEXT_PACK = _knob("TEXT_PACK", 1)      # 1 = pool consecutive documents into exactly-`text`-byte
                                       # samples. 0 = the old single-document window, which
                                       # DROPS every document shorter than text/2 and caps a
                                       # document at 4 samples - at text=1024/2048 that deletes
                                       # whole sources from the mix without saying so.
_TXT_BUF = {}                          # per-worker remainder, keyed by sample length


def _x_t_text(ex, spec, rng, key="text"):
    raw = (ex.get(key) or "").encode("utf-8", "ignore")
    L = spec.text
    out = []
    if not TEXT_PACK:
        if len(raw) < L // 2:
            return []
        n = min(4, max(1, len(raw) // L))
        for _ in range(n):
            st = rng.randint(0, max(0, len(raw) - L))
            a = np.zeros(L, np.uint8); c = raw[st:st + L]; a[:len(c)] = np.frombuffer(c, np.uint8)
            out.append(dict(t="text0", tgt=a))
        return out
    buf = _TXT_BUF.pop(L, b"") + raw + b"\n"
    while len(buf) >= L:
        out.append(dict(t="text0", tgt=np.frombuffer(buf[:L], np.uint8).copy()))
        buf = buf[L:]
    _TXT_BUF[L] = buf                     # shorter than L: carried to the next document
    return out

def x_t_text(ex, spec, rng): return _x_t_text(ex, spec, rng, "text")
def x_t_content(ex, spec, rng): return _x_t_text(ex, spec, rng, "content")


def _prompt_row(p, r, spec):
    p, r = (p or "").strip(), (r or "").strip()
    if len(p) < 4 or len(r) < 8:
        return []
    return [dict(t="text", cond=_bytes(p, spec.cond)[0], tgt=_bytes(r, spec.text)[0])]

def x_p_numina(ex, spec, rng): return _prompt_row(ex.get("problem"), ex.get("solution"), spec)

def x_p_smoltalk(ex, spec, rng):
    m = ex.get("messages") or []
    u = next((x["content"] for x in m if x.get("role") == "user"), None)
    a = next((x["content"] for x in m if x.get("role") == "assistant"), None)
    return _prompt_row(u, a, spec)


def _img_row(img, cap, spec):
    """img: {'bytes':..} from an undecoded Image column. JPEGs stay as bytes for nvJPEG; others are decoded here."""
    cap = _clean(cap)
    if not cap or not img or not img.get("bytes"):
        return []
    raw = img["bytes"]
    row = dict(t="image", cond=_bytes(cap, spec.cond)[0])
    if GPU_JPEG and HAS_NVJPEG and raw[:2] == b"\xff\xd8":
        row["jpeg"] = raw
        return [row]
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    S = spec.side
    im.draft("RGB", (S * 2, S * 2))
    im = im.convert("RGB")
    w, h = im.size
    s = S / min(w, h)
    im = im.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - S) // 2, (h - S) // 2
    row["img"] = np.asarray(im.crop((l, t, l + S, t + S)), dtype=np.uint8).transpose(2, 0, 1).copy()
    return [row]

def x_i_wds(ex, spec, rng): return _img_row(ex.get("jpg"), ex.get("txt"), spec)
def x_i_coco(ex, spec, rng):
    a = ex.get("answer")
    return _img_row(ex.get("image"), a[0] if isinstance(a, list) and a else a, spec)
def x_i_docvqa(ex, spec, rng):
    a = ex.get("answers") or [""]
    return _img_row(ex.get("image"), f"{ex.get('question', '')} {a[0] if a else ''}", spec)


def _wave(aud, spec):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(1)
    if sr != spec.sr:
        x = torch.from_numpy(data)[None, None]
        data = F.interpolate(x, size=int(len(data) * spec.sr / sr), mode="linear", align_corners=False)[0, 0].numpy()
    return data


_AUD = {"seen": 0, "kept": 0}      # audio clips offered vs kept, per worker process


def _aud_row(aud, text, spec):
    text = _clean(text).lower()
    if not text or not aud or not aud.get("bytes"):
        return []
    w = _wave(aud, spec)
    n = spec.wave_len
    _AUD["seen"] += 1
    if len(w) > n or len(w) < spec.sr // 2:           # a crop would no longer match the transcript
        if _AUD["seen"] % 500 == 1:
            print(f"  [audio] {_AUD['seen']} clips offered, {_AUD['kept']} kept "
                  f"({100 * _AUD['kept'] / max(1, _AUD['seen']):.0f}%) at a {n / spec.sr:.1f} s window "
                  f"- raise NCM_AUDIO_LEN if this is low", file=sys.stderr, flush=True)
        return []
    _AUD["kept"] += 1
    pcm = np.zeros(n, np.int16)
    pcm[:len(w)] = np.clip(w * 32767, -32768, 32767).astype(np.int16)
    return [dict(t="audio", cond=_bytes(text, spec.cond)[0], wav=pcm)]

def x_a_text(ex, spec, rng): return _aud_row(ex.get("audio"), ex.get("text"), spec)
def x_a_tts(ex, spec, rng): return _aud_row(ex.get("audio"), ex.get("text_normalized") or ex.get("text_original"), spec)

EXTRACT = {k[2:]: v for k, v in list(globals().items()) if k.startswith("x_")}


def open_stream(src, part, nparts, seed):
    from datasets import load_dataset, Audio, Image
    kw = dict(split=src.split, streaming=True)
    ds = None
    if STREAM_PREFETCH:                              # Parquet read-ahead + 128 MiB requests (HF "streaming datasets" blog)
        try:
            import pyarrow, pyarrow.dataset as pds
            fso = pds.ParquetFragmentScanOptions(cache_options=pyarrow.CacheOptions(prefetch_limit=1, range_size_limit=128 << 20))
            ds = (load_dataset(src.path, src.config, fragment_scan_options=fso, **kw) if src.config
                  else load_dataset(src.path, fragment_scan_options=fso, **kw))
        except Exception:
            ds = None                                # non-Parquet builder or older `datasets`: plain stream
    if ds is None:
        ds = load_dataset(src.path, src.config, **kw) if src.config else load_dataset(src.path, **kw)
    if part == 1 and nparts > 1:
        print(f"    [{src.name}] shards={ds.n_shards} (streams {nparts - 1})", flush=True)
    for col, kind in (src.raw or {}).items():
        ds = ds.cast_column(col, Audio(decode=False) if kind == "audio" else Image(decode=False))
    if nparts > 1:
        n_sh = ds.n_shards or 1
        if n_sh >= nparts:
            ds = ds.shard(num_shards=nparts, index=part)
        elif n_sh >= 2:                              # fewer shards than parts: part 0 (validation) owns shard 0, the training workers share shards 1.. (two workers may read one shard, each with its own shuffle)
            ds = ds.shard(num_shards=n_sh, index=0 if part == 0 else 1 + (part - 1) % (n_sh - 1))
        else:                                        # one shard: partition by row index (every part reads it all)
            ds = ds.filter(lambda ex, i, _p=part, _n=nparts: i % _n == _p, with_indices=True)
    return iter(ds.shuffle(seed=seed, buffer_size=src.buf))


def rows_of(src, spec, part, nparts, seed, rng, log=None):
    """Endless generator of rows from one source; reconnects with backoff, gives up after repeated failure."""
    fn, fails, epoch = EXTRACT[src.fn], 0, 0
    while True:
        try:
            for ex in open_stream(src, part, nparts, seed + 977 * epoch):
                for r in fn(ex, spec, rng):
                    r["src"] = src.name
                    fails = 0
                    yield r
            epoch += 1                                # stream exhausted: reshuffle and go again
        except Exception as e:
            fails += 1
            if log:
                import traceback
                fr = traceback.extract_tb(e.__traceback__)[-1]
                log(f"[{src.name}] {type(e).__name__}: {str(e)[:100]} @ {os.path.basename(fr.filename)}:{fr.lineno} (fail {fails})")
            if fails > 6:
                if log:
                    log(f"[{src.name}] dropped")
                return
            time.sleep(min(30, 2 ** fails) * (0.5 + random.random()))      # jitter: workers must not retry in lock-step


# ---- row-type mix: codes share per modality -> sampling probability per row type ----
CODE_SHARE = {"text": 0.40, "image": 0.35, "audio": 0.25}      # of live codes
TEXT_PROMPT_FRAC = 0.15                                          # of text codes that are prompt->response
def type_probs():
    share = {"text0": CODE_SHARE["text"] * (1 - TEXT_PROMPT_FRAC), "text": CODE_SHARE["text"] * TEXT_PROMPT_FRAC,
             "image": CODE_SHARE["image"], "audio": CODE_SHARE["audio"]}
    p = {r: share[r] / LAYOUTS[r].Tt for r in ROW_TYPES}                # codes -> rows
    z = sum(p.values())
    return {r: v / z for r, v in p.items()}
TYPE_P = type_probs()


class MixDataset:
    """Samples sources by (row-type probability x source weight) and yields raw rows for one worker."""

    def __init__(self, spec, kinds=None, part0=1, nparts=2, seed=0, even=False):
        self.spec, self.part0, self.nparts, self.seed = spec, part0, nparts, seed
        srcs = [s for s in SOURCES if kinds is None or s.kind in kinds]
        tw = {}
        for s in srcs:
            tw[s.kind] = tw.get(s.kind, 0) + s.weight
        self.srcs = srcs
        self.p = [(1.0 if even else TYPE_P[s.kind]) * s.weight / tw[s.kind] for s in srcs]   # even: equal share per row type

    def rows(self, wid=0):
        part = self.part0 + wid                      # worker w reads partition part0 + w
        rng = random.Random(self.seed * 7919 + part)
        log = lambda m: print(f"  [stream w{wid}] {m}", file=sys.stderr, flush=True)
        gens = {}
        alive = list(range(len(self.srcs)))
        while alive:
            i = rng.choices(alive, [self.p[j] for j in alive])[0]
            if i not in gens:
                gens[i] = rows_of(self.srcs[i], self.spec, part, self.nparts, self.seed * 31 + part, rng, log)
            try:
                yield next(gens[i])
            except StopIteration:
                alive.remove(i)


def _stream_worker(ds, wid, q, stop):
    torch.set_num_threads(1)                         # a forked child must not inherit the parent's OpenMP pool
    time.sleep(min(20.0, 0.6 * wid))                 # stagger the start: dozens of workers resolving the same Hub datasets at once get rate-limited
    try:
        for row in ds.rows(wid):
            while not stop.is_set():
                try:
                    q.put(row, timeout=1.0)
                    break
                except queue.Full:
                    pass
            if stop.is_set():
                return
    except Exception as e:
        print(f"  [stream w{wid}] worker died: {e!r}", file=sys.stderr, flush=True)


class RowStream:
    """Rows from `workers` forked processes into one queue, delivered in ARRIVAL order.

    (A torch DataLoader hands out worker batches strictly round-robin, so one worker that is busy
    opening a slow dataset stalls every other worker. Streaming needs the opposite.)
    """

    def __init__(self, ds, workers, depth=256):
        self.ds, self.workers = ds, workers
        if workers == 0:
            self.gen = ds.rows(0)
            return
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        self.q, self.stop = ctx.Queue(maxsize=depth), ctx.Event()
        self.procs = [ctx.Process(target=_stream_worker, args=(ds, w, self.q, self.stop), daemon=True)
                      for w in range(workers)]
        for pr in self.procs:
            pr.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self.workers == 0:
            return next(self.gen)
        while True:
            try:
                return self.q.get(timeout=2.0)
            except queue.Empty:
                if not any(pr.is_alive() for pr in self.procs):
                    raise RuntimeError("all stream workers exited")

    def close(self):
        if self.workers:
            self.stop.set()
            time.sleep(0.2)
            for pr in self.procs:
                if pr.is_alive():
                    pr.terminate()


# ---- rows -> GPU codes ----
def micro_batch_sizes():
    return {r: max(1, MICRO_TOKENS // LAYOUTS[r].T) for r in ROW_TYPES}
MICRO_B = micro_batch_sizes()


@torch.no_grad()
def encode_rows(rtype, rows, codecs, device=dev):
    """list of raw rows (one row type) -> dict(rtype, cond [B,Lc] long|None, codes list, n_live)."""
    B = len(rows)
    cond = None
    if LAYOUTS[rtype].cond:
        cond = torch.from_numpy(np.stack([r["cond"] for r in rows])).to(device).long() + 1
    if ROW_MOD[rtype] == "text":
        codes = codecs.text.encode(torch.from_numpy(np.stack([r["tgt"] for r in rows])).to(device))
    elif ROW_MOD[rtype] == "image":
        S = SPEC.side
        imgs = [None] * B
        jidx = [i for i, r in enumerate(rows) if "jpeg" in r]
        if jidx:
            dec = decode_jpeg_gpu([rows[i]["jpeg"] for i in jidx], S, device)
            for n, i in enumerate(jidx):
                imgs[i] = dec[n]
        for i, r in enumerate(rows):
            if imgs[i] is None:
                imgs[i] = torch.from_numpy(r["img"]).to(device)
        codes = codecs.image.encode(torch.stack(imgs))
    else:
        w = torch.from_numpy(np.stack([r["wav"] for r in rows])).to(device).float() / 32768.0
        codes = codecs.audio.encode(w)
    return dict(rtype=rtype, cond=cond, codes=codes, n_live=B * LAYOUTS[rtype].Tt,
                src=[r.get("src", "?") for r in rows])


class Prefetcher:
    """Background thread: loader rows -> per-type micro-batches -> GPU codecs (side stream) -> RAM ring."""

    def __init__(self, loader, codecs, depth=PREFETCH, writer=None):
        self.q = queue.Queue(maxsize=depth)
        self.writer = writer
        self.loader, self.codecs = loader, codecs   # loader: any iterator of raw rows (RowStream)
        self.stop = threading.Event()
        self.err = None
        self.rows_in = 0
        self.stream = torch.cuda.Stream() if IS_CUDA else None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            bufs = {r: [] for r in ROW_TYPES}
            for row in self.loader:
                if self.stop.is_set():
                    return
                self.rows_in += 1
                b = bufs[row["t"]]
                b.append(row)
                if len(b) >= MICRO_B[row["t"]]:
                    rows, bufs[row["t"]] = b, []
                    if self.stream is not None:
                        with torch.cuda.stream(self.stream):
                            mb = encode_rows(row["t"], rows, self.codecs)
                            if self.writer is not None:
                                self.writer.add(mb)             # D2H copy on the side stream: never waits on training kernels
                        self.stream.synchronize()
                    else:
                        mb = encode_rows(row["t"], rows, self.codecs)
                        if self.writer is not None:
                            self.writer.add(mb)
                    while not self.stop.is_set():
                        try:
                            self.q.put(mb, timeout=1.0)
                            break
                        except queue.Full:
                            pass
        except Exception as e:                                  # surfaced to the trainer instead of a silent stall
            self.err = e

    def get(self, timeout=600):
        t0 = time.time()
        while True:
            try:
                mb = self.q.get(timeout=1.0)
                break
            except queue.Empty:
                if self.err is not None:
                    raise RuntimeError(f"data thread died: {self.err!r}")
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"no batch for {timeout}s: streams are stalled or throttled (set HF_TOKEN)")
        if IS_CUDA:                                              # tensors were born on the side stream
            cur = torch.cuda.current_stream()
            for t in ([mb["cond"]] if mb["cond"] is not None else []) + mb["codes"]:
                t.record_stream(cur)
        return mb

    def close(self):
        self.stop.set()


def collect_rows(kinds, quota, part0, nparts, workers, timeout=1800, seed=0, even=False):
    """Pull `quota[row_type]` raw rows into RAM from partitions part0.. and stop. Nothing touches disk."""
    ds = MixDataset(SPEC, kinds=kinds, part0=part0, nparts=nparts, seed=seed, even=even)
    out, t0, last = [], time.time(), time.time()
    cnt = {k: 0 for k in quota}
    stream = RowStream(ds, workers)
    try:
        for row in stream:
            t = row["t"]
            if t in cnt and cnt[t] < quota[t]:
                out.append(row); cnt[t] += 1
            if all(cnt[k] >= quota[k] for k in cnt):
                break
            if time.time() - last > 20:
                print(f"    collecting {cnt} / {quota}  ({time.time() - t0:.0f}s)", flush=True); last = time.time()
            if time.time() - t0 > timeout:
                print(f"    !! timeout: got {cnt} of {quota}")
                break
    finally:
        stream.close()
    return out
