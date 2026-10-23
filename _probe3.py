"""Probe 3: hf:// -> local file for streaming video, then decode with av."""
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import av
from datasets import features, load_dataset
from huggingface_hub import hf_hub_download


def hf_to_local(uri):
    body = uri[len("hf://datasets/"):]
    repo, rest = body.split("@", 1)
    rev, fname = rest.split("/", 1)
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", revision=rev)


def frames(local, k=8):
    c = av.open(local, mode="r")
    out = []
    for f in c.decode(video=0):
        out.append(f.to_image())
        if len(out) >= k:
            break
    c.close()
    return out


ds = load_dataset("sayakpaul/ucf101-subset", split="train", streaming=True)
ds = ds.cast_column("video", features.Video(decode=False))
for i, ex in enumerate(ds):
    uri = ex["video"]["path"]
    local = hf_to_local(uri)
    fr = frames(local)
    print(f"[{i}] {os.path.basename(local)} -> {len(fr)} frames, first {fr[0].size} {fr[0].mode}")
    if i >= 1:
        break
