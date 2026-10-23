"""Reproduce: does a real cifar sample survive _img_cap -> encode_sample?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni.producer import SOURCES, Source, encode_sample, make_codecs
from omni.schedule import OmniSchedule

sched = OmniSchedule(base_len=dict(text=64, image=64, audio=96, video=32), K=6, R=2)
codecs = make_codecs(sched, 256, True)

for name in ("image_cifar10", "audio_libri", "video_ucf"):
    try:
        src = Source(name, SOURCES[name][0], SOURCES[name][1])
        batch = src.batch(2)
        print(f"\n=== {name}: {len(batch)} samples")
        for i, parts in enumerate(batch):
            print(f"  keys={sorted(parts)}", end="  ")
            for k, v in parts.items():
                if k == "text":
                    print(f"text={str(v)[:28]!r}", end=" ")
                elif k == "audio":
                    print(f"audio={type(v).__name__}", end=" ")
                else:
                    print(f"{k}={type(v).__name__}", end=" ")
            codes, present = encode_sample(codecs, sched, parts)
            print(f"-> present={present} sizes={ {k: len(v) for k, v in codes.items()} }")
            if i >= 1:
                break
    except Exception as e:
        print(f"\n=== {name}: FAIL {type(e).__name__}: {str(e)[:200]}")
