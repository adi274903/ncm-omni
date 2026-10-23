"""Which streaming video datasets actually yield clips? Cast to raw, count what we can decode."""
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from datasets import features, load_dataset

CANDIDATES = [
    "sayakpaul/ucf101-subset",
    "nateraw/kinetics-mini",
    "hf-internal-testing/small-video-dataset",
    "HuggingFaceM4/webvid",
    "juliensimon/kinetics-400",
]

for cid in CANDIDATES:
    try:
        ds = load_dataset(cid, split="train", streaming=True)
        info = ds.features
        ds2 = ds.cast_column("video", features.Video(decode=False))
        n = 0
        for _ in ds2:
            n += 1
            if n >= 12:
                break
        print(f"[{cid}] OK rows>= {n} | fast moves to next")
    except Exception as e:
        print(f"[{cid}] FAIL {type(e).__name__}: {str(e)[:120]}")
