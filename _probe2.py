"""Probe 2: raw-bytes paths for audio (soundfile) and video (av)."""
import io
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import av
import numpy as np
import soundfile as sf
from datasets import features, load_dataset


def audio_probe():
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    ds = ds.cast_column("audio", features.Audio(decode=False))
    ex = next(iter(ds))
    a = ex["audio"]
    keys = sorted(a.keys()) if isinstance(a, dict) else str(a)[:60]
    raw = a.get("bytes") if isinstance(a, dict) else None
    path = a.get("path") if isinstance(a, dict) else None
    if raw is None and path:
        raw = open(path, "rb").read()
    sr = None
    if raw:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return keys, len(raw) if raw else None, sr, (np.asarray(wav)[:, 0].shape if raw else None)


def video_probe():
    ds = load_dataset("sayakpaul/ucf101-subset", split="train", streaming=True)
    ds = ds.cast_column("video", features.Video(decode=False))
    ex = next(iter(ds))
    v = ex["video"]
    path = v.get("path") if isinstance(v, dict) else None
    raw = v.get("bytes") if isinstance(v, dict) else None
    src = io.BytesIO(raw) if raw else path
    c = av.open(src, mode="r")
    st = c.streams.video[0]
    frames = []
    for f in c.decode(video=0):
        frames.append(f.to_image())
        if len(frames) >= 8:
            break
    c.close()
    return path is not None, raw is not None, len(frames), frames[0].size if frames else None


for tag, fn in (("audio_raw", audio_probe), ("video_raw", video_probe)):
    try:
        print(f"[{tag}] OK {fn()}")
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:200]}")
