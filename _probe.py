"""Probe HF dataset ids for the producer's SOURCES table. Read-only, no writes to run/."""
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from datasets import features, load_dataset


def probe(tag, fn):
    try:
        print(f"[{tag}] OK {fn()}")
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:160]}")


probe("cifar10", lambda: list(next(iter(load_dataset("uoft-cs/cifar10", split="train", streaming=True))).keys()))
probe("cifar100", lambda: list(next(iter(load_dataset("uoft-cs/cifar100", split="train", streaming=True))).keys()))
probe("libri_splits", lambda: list(load_dataset("openslr/librispeech_asr", "clean", streaming=True).keys()))
probe("libri_tr100", lambda: list(next(iter(load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True))).keys()))
probe("ucf_fields", lambda: list(next(iter(load_dataset("sayakpaul/ucf101-subset", split="train", streaming=True))).keys()))


def ucf_cast():
    ds = load_dataset("sayakpaul/ucf101-subset", split="train", streaming=True)
    ds = ds.cast_column("video", features.Video(decode=False))
    ex = next(iter(ds))
    v = ex["video"]
    if isinstance(v, dict):
        return sorted(v.keys()), (len(v["bytes"]) if v.get("bytes") else None)
    return type(v).__name__, str(v)[:60]


probe("ucf_cast", ucf_cast)
