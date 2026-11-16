# %% [markdown]
# ## 6. Codec fit and held-out set (streamed, held in RAM only; stored under `SHARD_DIR` when shards are on)
#
# This stage needs no model, so it sits right after the data and shard sections. With `SHARD_MODE="encode"` the notebook
# goes from here straight to encoding the dataset (next cell); the kernel, model and build sections further down are
# skipped, so no GPU time is spent on them.

# %%
if not META_ONLY or SHARD_MODE == "encode":
    PARTS = max(WORKERS, 1) + 1                 # part 0 = validation; parts 1.. = training workers
    codecs = Codecs(SPEC, dev)

    assert not (TOKENIZER in ("vq", "feat") and USE_SHARDS), "shards store codes of one fixed codec: use TOKENIZER='kmeans' with SHARD_MODE record/read/encode, or SHARD_MODE='off'"

    def fit_codecs():
        if TOKENIZER == "feat":
            return fit_feature_codecs()                   # frozen DINOv2 / DAC features (section 3b): fit the codebooks, no training
        if TOKENIZER == "vq":
            return fit_tokenizers()                       # learned tokenizers (previous section): train, compare with k-means, gate
        t0 = time.time()
        fdir = os.path.join(SHARD_DIR, "fit") if USE_SHARDS else ""         # per-modality sample sets: raw enough to refit, tiny next to the raw data
        f_img, f_aud = os.path.join(fdir, "image.npy"), os.path.join(fdir, "audio.npy")
        feats, B = [], 64
        if fdir and os.path.exists(f_img):
            imgs_u8 = np.load(f_img, mmap_mode="r")
            print(f"fit set for images from {f_img} ({len(imgs_u8)} images, no streaming)")
            n_b = len(imgs_u8) // B + 1
            for i in range(0, len(imgs_u8), B):
                feats.append(sample_points(codecs.image.features(torch.from_numpy(np.ascontiguousarray(imgs_u8[i:i + B])).to(dev)), 200_000 // n_b))
        else:
            print(f"fitting frozen codebooks from {FIT_IMAGES} streamed images and {FIT_AUDIO} audio clips (RAM only)...")
            rows = collect_rows(("image",), {"image": FIT_IMAGES}, 1, PARTS, WORKERS)
            keep = []
            for i in range(0, len(rows), B):
                sub = rows[i:i + B]
                tmp = [None] * len(sub)
                jp = [j for j, r in enumerate(sub) if "jpeg" in r]
                if jp:
                    d = decode_jpeg_gpu([sub[j]["jpeg"] for j in jp], SPEC.side, dev)
                    for n, j in enumerate(jp):
                        tmp[j] = d[n]
                for j, r in enumerate(sub):
                    if tmp[j] is None:
                        tmp[j] = torch.from_numpy(r["img"]).to(dev)
                batch = torch.stack(tmp)
                if fdir:
                    keep.append(batch.cpu().numpy())
                feats.append(sample_points(codecs.image.features(batch), 200_000 // max(1, len(rows) // B + 1)))
            if fdir and keep:
                os.makedirs(fdir, exist_ok=True)
                np.save(f_img, np.concatenate(keep))
            del rows, keep
        codecs.image.fit(feats); del feats
        print(f"  image codebooks done ({time.time() - t0:.0f}s)")
        if fdir and os.path.exists(f_aud):
            wav_u8 = np.load(f_aud, mmap_mode="r")
            print(f"fit set for audio from {f_aud} ({len(wav_u8)} clips, no streaming)")
            waves = [torch.from_numpy(np.ascontiguousarray(wav_u8[i:i + 32])).to(dev).float() / 32768 for i in range(0, len(wav_u8), 32)]
        else:
            rows = collect_rows(("audio",), {"audio": FIT_AUDIO}, 1, PARTS, WORKERS)
            if fdir and rows:
                os.makedirs(fdir, exist_ok=True)
                np.save(f_aud, np.stack([r["wav"] for r in rows]))
            waves = [torch.from_numpy(np.stack([r["wav"] for r in rows[i:i + 32]])).to(dev).float() / 32768 for i in range(0, len(rows), 32)]
            del rows
        codecs.audio.set_stats(waves)
        codecs.audio.fit([sample_points(codecs.audio.features(w), 200_000 // max(1, len(waves))) for w in waves])
        print(f"  audio codebooks done ({time.time() - t0:.0f}s)  mel mean {codecs.audio.mean:.2f} std {codecs.audio.std:.2f}")

    CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt") if CKPT_DIR else ""
    RESUME = None
    if CKPT_PATH and os.path.exists(CKPT_PATH):
        RESUME = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        _ci_state = RESUME["codecs"]["image"]
        _kind = _ci_state["kind"] if isinstance(_ci_state, dict) and _ci_state.get("kind") in ("vq", "feat") else "kmeans"
        _want = "vq" if (TOKENIZER == "vq" or (TOKENIZER == "feat" and IMAGE_CODEC == "vq")) else TOKENIZER     # the image codec's state kind
        if _kind != _want:                                # e.g. the old 2B checkpoint in the same Drive folder: never mix codecs
            print(f"!! {CKPT_PATH} was trained with the '{_kind}' image codec but this run wants '{_want}' (TOKENIZER='{TOKENIZER}', IMAGE_CODEC='{IMAGE_CODEC}'): ignoring it (use a new CKPT_DIR to keep both)")
            RESUME = None
    if RESUME is not None:
        codecs.load(RESUME["codecs"])
        print(f"resuming from {CKPT_PATH}: step {RESUME['step']}, codebooks restored (no refit)")
    elif SHARD_DIR and SHARD_MODE in ("record", "read", "encode") and os.path.exists(os.path.join(SHARD_DIR, "codecs.pt")):
        codecs.load(torch.load(os.path.join(SHARD_DIR, "codecs.pt"), map_location="cpu", weights_only=False))
        print(f"codebooks restored from {SHARD_DIR} (the shards were encoded with them; no refit)")
    elif SHARD_MODE == "read":
        raise RuntimeError(f"SHARD_MODE='read' but {SHARD_DIR}/codecs.pt does not exist: run once with SHARD_MODE='record' or 'encode'.")
    else:
        fit_codecs()
    if USE_SHARDS:
        print(f"shards: {SHARD_MODE} | dir {SHARD_DIR} | codebook fingerprint {shard_prepare(SHARD_DIR, codecs)}")

    def build_val():
        t0 = time.time()
        vdir = os.path.join(SHARD_DIR, "val") if USE_SHARDS else ""         # encoded held-out rows, one file per row type
        chunk = {r: min(MICRO_B[r], 16) for r in ROW_TYPES}
        if vdir and all(os.path.exists(os.path.join(vdir, f"{r}.npy")) for r in ROW_TYPES):
            out = []
            for r in ROW_TYPES:
                arr = np.load(os.path.join(vdir, f"{r}.npy"))
                for i in range(0, len(arr), chunk[r]):
                    cond, codes = unpack_rows(r, arr[i:i + chunk[r]], dev)
                    n = len(arr[i:i + chunk[r]])
                    out.append(dict(rtype=r, cond=cond, codes=codes, n_live=n * LAYOUTS[r].Tt, src=["val"] * n))
            print(f"held-out set from {vdir}: { {r: sum(len(b['src']) for b in out if b['rtype'] == r) for r in ROW_TYPES} } rows, no streaming")
            return out
        quota = {r: VAL_ROWS for r in ROW_TYPES}
        rows = collect_rows(None, quota, 0, PARTS, 1, seed=12345, even=True)
        out = []
        for r in ROW_TYPES:
            rr = [x for x in rows if x["t"] == r]
            for i in range(0, len(rr), chunk[r]):
                out.append(encode_rows(r, rr[i:i + chunk[r]], codecs))
        if vdir and all(any(b["rtype"] == r for b in out) for r in ROW_TYPES):
            os.makedirs(vdir, exist_ok=True)
            for r in ROW_TYPES:
                np.save(os.path.join(vdir, f"{r}.npy"), np.concatenate([pack_rows(b) for b in out if b["rtype"] == r]))
        print(f"held-out set: { {r: sum(len(b['src']) for b in out if b['rtype'] == r) for r in ROW_TYPES} } rows ({time.time() - t0:.0f}s)")
        return out

    VAL = build_val()
