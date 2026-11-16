# %% [markdown]
# ## 5. Shards: encode once, keep exactly what the model reads (the shard maker is `encode_shards()` at the end of the next cell)
#
# The model never sees raw rows, only the **codes**: one `uint8` per code (code - 1, so 1..256 fits in a byte),
# stored per row as `[ prefix | level 0 | level 1 | ... | level K-1 ]`. That is ~300x smaller than the raw
# images and audio (a 341-byte row instead of a ~100 KB JPEG), so 2B codes is ~2 GB on disk.
#
# | `SHARD_MODE` | what happens |
# |---|---|
# | `encode` | **no model, no training**: stream the whole dataset (`SHARD_CODES_B` / `TOKENS` / the tier), encode it on the GPU and store it, one pass per row type, resumable. Run the notebook once like this, then switch to `read` |
# | `record` (default) | train from the live stream as before, and **write every encoded micro-batch to `SHARD_DIR`** |
# | `read` | train **only** from the stored shards (no Hub streaming for training; exact, seeded, resumable data order) |
# | `off` | the previous behaviour: stream, store nothing |
#
# Shards are only valid for the **codebooks they were encoded with**, so the codebooks are saved next to them
# (`codecs.pt`) with a fingerprint. A later run loads those codebooks instead of refitting, and refuses to mix
# shards with different ones. The same directory also keeps the **codebook-fit samples** (`fit/image.npy`, `fit/audio.npy`)
# and the **held-out set** (`val/<row type>.npy`, from partition 0, so it never overlaps the shards), so a new session
# in the same runtime/disk does not re-stream them.

# %%
import hashlib
from glob import glob as _glob

SHARD_MB = _knob("SHARD_MB", 64)                                       # target size of one shard file
ROW_W = {r: LAYOUTS[r].cond + LAYOUTS[r].Tt for r in ROW_TYPES}        # stored bytes per row: prefix + every level
USE_SHARDS = bool(SHARD_DIR) and SHARD_MODE in ("record", "read", "encode")


def codec_fingerprint(cd):
    """Hash of everything the codes depend on: both codebooks, the mel statistics and the ladder spec."""
    h = hashlib.sha1()
    for cb in (cd.image.cb if cd.image.cb is not None else []) + (cd.audio.cb if cd.audio.cb is not None else []):
        h.update(cb.detach().float().cpu().numpy().tobytes())
    h.update(repr((round(cd.audio.mean, 6), round(cd.audio.std, 6))).encode())
    h.update(repr(SPEC).encode())
    return h.hexdigest()[:16]


def pack_rows(mb):
    """encoded micro-batch (long codes 1..256 on the device) -> uint8 numpy [B, ROW_W]."""
    parts = ([mb["cond"] - 1] if mb["cond"] is not None else []) + [c - 1 for c in mb["codes"]]
    return torch.cat(parts, 1).to(torch.uint8).cpu().numpy()


def unpack_rows(rtype, rows_u8, device):
    """uint8 numpy [B, ROW_W] -> (cond long [B, Lc] | None, codes: K long tensors), on `device`."""
    lay = LAYOUTS[rtype]
    t = torch.from_numpy(np.ascontiguousarray(rows_u8))
    if device.type == "cuda":
        t = t.pin_memory().to(device, non_blocking=True)
    cond = t[:, :lay.cond].long() + 1 if lay.cond else None
    codes, o = [], lay.cond
    for n in lay.lens:
        codes.append(t[:, o:o + n].long() + 1)
        o += n
    return cond, codes


def shard_prepare(root, cd):
    """Create/validate the shard directory against the codebooks that are about to be used."""
    os.makedirs(root, exist_ok=True)
    meta_p, fp = os.path.join(root, "meta.json"), codec_fingerprint(cd)
    if os.path.exists(meta_p):
        old = json.load(open(meta_p))["fingerprint"]
        if old != fp:
            raise RuntimeError(f"shards in {root} were encoded with other codebooks ({old} != {fp}). Use a new SHARD_DIR, "
                               f"delete this one, or resume from the checkpoint/shards that hold the matching codebooks.")
    elif SHARD_MODE in ("record", "encode"):
        json.dump(dict(fingerprint=fp, spec=repr(SPEC), preset=PRESET, row_w=ROW_W), open(meta_p, "w"))
        torch.save(cd.state(), os.path.join(root, "codecs.pt"))
    return fp


def shard_stats(root):
    n_files, n_bytes, rows = 0, 0, {}
    for r in ROW_TYPES:
        fs = _glob(os.path.join(root, r, "*.npy"))
        n_files += len(fs)
        n_bytes += sum(os.path.getsize(f) for f in fs)
        rows[r] = sum(np.load(f, mmap_mode="r").shape[0] for f in fs)
    return n_files, n_bytes, rows


class ShardWriter:
    """Appends encoded micro-batches to per-row-type shard files. Called from the prefetch thread."""

    def __init__(self, root):
        self.root, self.lock = root, threading.Lock()
        self.buf = {r: [] for r in ROW_TYPES}
        self.nbuf = {r: 0 for r in ROW_TYPES}
        self.written = {r: 0 for r in ROW_TYPES}
        self.idx = {}
        for r in ROW_TYPES:
            os.makedirs(os.path.join(root, r), exist_ok=True)
            have = [int(os.path.basename(f)[:-4]) for f in _glob(os.path.join(root, r, "*.npy"))]
            self.idx[r] = max(have) + 1 if have else 0

    def add(self, mb):
        r, a = mb["rtype"], pack_rows(mb)
        with self.lock:
            self.buf[r].append(a)
            self.nbuf[r] += len(a)
            if self.nbuf[r] * ROW_W[r] >= SHARD_MB * 2 ** 20:
                self._flush(r)

    def _flush(self, r):
        if not self.buf[r]:
            return
        arr = np.concatenate(self.buf[r])
        self.buf[r], self.nbuf[r] = [], 0
        path = os.path.join(self.root, r, f"{self.idx[r]:06d}.npy")
        self.idx[r] += 1
        with open(path + ".tmp", "wb") as f:
            np.save(f, arr)
        os.replace(path + ".tmp", path)                     # a crash never leaves a half-written shard
        self.written[r] += len(arr)

    def flush_all(self):
        with self.lock:
            for r in ROW_TYPES:
                self._flush(r)

    def close(self):
        self.flush_all()


class ShardFeeder(Prefetcher):
    """Same interface as `Prefetcher` (get / close), fed from stored shards instead of the stream.

    Micro-batch number m is drawn with a generator seeded by (SEED, m), so a resumed run (start at
    step * ACCUM) continues the exact same data order. Rows are sampled uniformly with replacement
    over all stored rows of the chosen row type, so a run longer than the data simply makes more epochs."""

    def __init__(self, root, start_micro=0, depth=PREFETCH, seed=0, micro_left=0):
        self.q = queue.Queue(maxsize=depth)
        self.stop, self.err, self.rows_in = threading.Event(), None, 0
        self.stream = torch.cuda.Stream() if IS_CUDA else None
        self.seed, self.m0 = seed, start_micro
        self.arr = {r: [np.load(f, mmap_mode="r") for f in sorted(_glob(os.path.join(root, r, "*.npy")))] for r in ROW_TYPES}
        self.cum = {r: np.cumsum([len(a) for a in self.arr[r]]) for r in ROW_TYPES}
        self.rows = {r: int(self.cum[r][-1]) if len(self.arr[r]) else 0 for r in ROW_TYPES}
        short = [r for r in ROW_TYPES if self.rows[r] < 1]
        if short:
            raise RuntimeError(f"no stored rows for {short} in {root}: {self.rows}. Run with SHARD_MODE='record' first.")
        p = np.array([TYPE_P[r] for r in ROW_TYPES])
        self.p = p / p.sum()
        epochs = {r: micro_left * self.p[i] * MICRO_B[r] / self.rows[r] for i, r in enumerate(ROW_TYPES)}
        print(f"reading shards: rows {self.rows} | this run draws about { {r: round(float(e), 2) for r, e in epochs.items()} } epochs of each", flush=True)
        if max(epochs.values()) > 4:
            print("  !! more than 4 epochs over some row type: record more data (longer live run) before reading", flush=True)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _batch(self, m):
        rng = np.random.default_rng([self.seed, m])
        r = ROW_TYPES[int(rng.choice(len(ROW_TYPES), p=self.p))]
        idx = np.sort(rng.integers(0, self.rows[r], size=MICRO_B[r]))
        fi = np.searchsorted(self.cum[r], idx, side="right")
        base = np.concatenate([[0], self.cum[r]])[fi]
        rows = np.concatenate([self.arr[r][f][(idx - base)[fi == f]] for f in np.unique(fi)])
        return r, rows

    def _run(self):
        try:
            m = self.m0
            while not self.stop.is_set():
                r, rows = self._batch(m)
                m += 1
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        cond, codes = unpack_rows(r, rows, dev)
                    self.stream.synchronize()
                else:
                    cond, codes = unpack_rows(r, rows, dev)
                mb = dict(rtype=r, cond=cond, codes=codes, n_live=len(rows) * LAYOUTS[r].Tt, src=["shard"] * len(rows))
                while not self.stop.is_set():
                    try:
                        self.q.put(mb, timeout=1.0)
                        break
                    except queue.Full:
                        pass
        except Exception as e:
            self.err = e


# ---- encode-only: build the whole dataset as shards before any training ----
SHARD_CODES_B = _knob("SHARD_CODES_B", 0.0)            # billions of live codes to store; 0 = TOKENS, else the tier
ENCODE_SHARE = {"text0": CODE_SHARE["text"] * (1 - TEXT_PROMPT_FRAC), "text": CODE_SHARE["text"] * TEXT_PROMPT_FRAC,
                "image": CODE_SHARE["image"], "audio": CODE_SHARE["audio"]}      # share of live codes per row type


def encode_shards(kinds=ROW_TYPES):
    """Stream, encode and store the dataset, one row type at a time. No model step. Resumable: what is already
    on disk is counted, and only the missing codes are streamed."""
    total = SHARD_CODES_B or TOKENS or {"T0": 0.02, "T1": 2.0, "T2": 20.0, "T3": 50.0}[TIER]
    os.makedirs(SHARD_DIR, exist_ok=True)
    free = shutil.disk_usage(SHARD_DIR).free / 1e9
    need_gb = total * 1.2
    print(f"encoding {total:.3f}B live codes (~{need_gb:.1f} GB on disk, {free:.0f} GB free) into {SHARD_DIR}")
    print("  per row type:", {r: f"{total * ENCODE_SHARE[r]:.3f}B" for r in ROW_TYPES}, flush=True)
    if need_gb > 0.95 * free:
        raise RuntimeError(f"not enough disk for the shards: need ~{need_gb:.0f} GB, {free:.0f} GB free. Point SHARD_DIR at Drive or lower SHARD_CODES_B.")
    writer = ShardWriter(SHARD_DIR)
    _, _, have_rows = shard_stats(SHARD_DIR)
    for r in kinds:
        quota, have = total * 1e9 * ENCODE_SHARE[r], have_rows[r] * LAYOUTS[r].Tt
        if have >= quota:
            print(f"  {r:6s} already stored: {have / 1e9:.3f}B of {quota / 1e9:.3f}B codes, skipped", flush=True)
            continue
        stream = RowStream(MixDataset(SPEC, kinds=(r,), part0=1, nparts=PARTS, seed=SEED + 10007 + have_rows[r] % 1_000_003), WORKERS)
        pf = Prefetcher(stream, codecs, writer=writer)
        got, t0, last = 0, time.time(), time.time()
        try:
            while have + got < quota:
                got += pf.get()["n_live"]
                if time.time() - last > 30:
                    rate = got / (time.time() - t0)
                    print(f"  {r:6s} {(have + got) / 1e9:.3f} / {quota / 1e9:.3f}B codes | {rate / 1e3:.1f}k codes/s | "
                          f"eta {(quota - have - got) / max(rate, 1) / 3600:.1f} h", flush=True)
                    last = time.time()
                if SESSION_HOURS and time.time() - T_START > SESSION_HOURS * 3600:
                    print(f"  watchdog: {SESSION_HOURS} h reached. Shards written so far are kept: run this cell again to continue.")
                    return False
        finally:
            pf.close(); stream.close(); writer.flush_all()
        print(f"  {r:6s} done: {(have + got) / 1e9:.3f}B codes in {(time.time() - t0) / 60:.1f} min", flush=True)
    writer.close()
    n_files, n_bytes, rows = shard_stats(SHARD_DIR)
    print(f"shards complete: {n_files} files, {n_bytes / 2**30:.2f} GiB, rows {rows}")
    return True
