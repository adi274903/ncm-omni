"""Probe real dataset candidates for the omni producer: does it stream, what columns, does the
first row survive extraction? Read-only. Reports one line per candidate."""
import io
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from datasets import features, load_dataset

TEXT_KEYS = ("text", "content", "raw_content", "document", "problem", "question", "prompt")
IMG_KEYS = ("image", "img", "png", "jpg")
CAP_KEYS = ("text", "caption", "captions", "alt_text", "label", "question", "problem")


def open_stream(cid, config, split):
    ds = load_dataset(cid, config, split=split, streaming=True)
    cols = set(ds.features)
    if "audio" in cols:
        ds = ds.cast_column("audio", features.Audio(decode=False))
    if "video" in cols:
        ds = ds.cast_column("video", features.Video(decode=False))
    return ds, cols, config, split


def probe(tag, cid, config=None, split="train"):
    t0 = time.time()
    try:
        ds, cols, cfg, sp = open_stream(cid, config, split)
        ex = next(iter(ds))
    except Exception as e:
        print(f"{tag:16s} FAIL {type(e).__name__}: {str(e)[:110]}")
        return
    dt = time.time() - t0
    note = []
    if "audio" in cols:
        a = ex["audio"]
        raw = None
        if isinstance(a, dict):
            raw = a.get("bytes")
            if not raw and a.get("path"):
                raw = open(a["path"], "rb").read() if os.path.exists(a["path"]) else None
        if raw:
            import soundfile as sf
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            note.append(f"audio {len(wav)}samples@{sr}")
        else:
            note.append(f"audio NORAW {type(a).__name__}")
    if "image" in cols or "img" in cols:
        k = "image" if "image" in cols else "img"
        v = ex[k]
        note.append(f"{k}={type(v).__name__}{getattr(v, 'size', '')}")
    if "video" in cols:
        v = ex["video"]
        note.append(f"video keys={sorted(v) if isinstance(v, dict) else type(v).__name__}")
    tk = next((k for k in TEXT_KEYS if isinstance(ex.get(k), str) and ex[k].strip()), None)
    if tk:
        note.append(f"text[{tk}] len={len(ex[tk])}")
    print(f"{tag:16s} OK  {dt:4.1f}s cfg={cfg} split={sp} cols={sorted(cols)}\n"
          f"{'':16s}     {', '.join(note) or 'no text/media found'}")


print("=== TEXT")
for tag, cid, cfg, split in [
    ("fineweb", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    ("smollm-python", "HuggingFaceTB/smollm-corpus", "python-edu", "train"),
    ("smollm-cosmo", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train"),
    ("openwebmath", "open-web-math/open-web-math", None, "train"),
    ("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train"),
    ("numina-cot", "AI-MO/NuminaMath-CoT", None, "train"),
    ("stack-smol-py", "bigcode/the-stack-smol", "data/python", "train"),
    ("pes2o", "allenai/peS2o", None, "train"),
    ("arxiv-sum", "ccdv/arxiv-summarization", None, "train"),
    ("openr1-math", "open-r1/OpenR1-Math-220k", None, "train"),
    ("wikipedia", "wikimedia/wikipedia", "20231101.en", "train"),
    ("openmath2", "nvidia/OpenMathReasoning", None, "train"),
]:
    probe(tag, cid, cfg, split)

print("\n=== VISION")
for tag, cid, cfg, split in [
    ("docvqa", "HuggingFaceM4/DocumentVQA", None, "train"),
    ("flowers", "nelorth/oxford-flowers", None, "train"),
    ("food101", "huggingface/food101", None, "train"),
    ("docci", "google/docci", None, "train"),
    ("coco30", "sayakpaul/coco-30-val-2014", None, "train"),
    ("vqav2-sm", "merve/vqav2-small", None, "train"),
    ("textvqa", "facebook/textvqa", None, "train"),
    ("cifar10", "uoft-cs/cifar10", None, "train"),
]:
    probe(tag, cid, cfg, split)

print("\n=== AUDIO")
for tag, cid, cfg, split in [
    ("libri100", "openslr/librispeech_asr", "clean", "train.100"),
    ("libritts-r", "mythicinfinity/libritts_r", "clean", "train.clean.100"),
    ("fleurs", "google/fleurs", "en_us", "train"),
    ("peoples", "MLCommons/peoples_speech", "microset", "train"),
    ("voxpopuli", "facebook/voxpopuli", "en", "train"),
    ("tedlium", "LIUM/tedlium", "release3", "train"),
    ("ears", "diabolico/ears", None, "train"),
]:
    probe(tag, cid, cfg, split)
