# %% [markdown]
# # NCM-MoE Omni: Next Concept Modelling, 6 modality experts, text / image / audio
#
# A from-scratch notebook (nothing imported from a repo). One transformer predicts **every code of a
# level in a single forward**, coarse to fine: `K` forwards generate a sample instead of one per
# token. Text, image and audio share one **level ladder** (level *k* has the same compression ratio
# `R^(K-1-k)` in every modality), so "level 2" means the same altitude everywhere.
#
# | piece | what it is |
# |---|---|
# | trunk | d=1536, L=16, GQA 16/4, QK-norm, block-causal-over-levels attention |
# | experts | 3 modality groups x 2 SwiGLU experts = **6 experts, 1.66B total, ~360M active** |
# | routing | group forced by the token's modality; inside the group, **expert-choice** (each expert takes its top-capacity tokens per level), so shapes are static, nothing is dropped from the graph, and a token gets 0, 1 or 2 experts (avg 1) |
# | conditioning | a text prefix: text->text (prompt), text->image (caption), text->speech (transcript = TTS) |
# | data | streamed straight from the Hub (HF cache lives in RAM). Every encoded row is also **recorded as `uint8` code shards on local disk** (`SHARD_DIR`, ~1 byte per code) so a later run can read them instead of streaming |
# | default run | **T2 = 20B live codes** (see the tier table in the budget cell) |
#
# Verification knobs: every knob can be overridden by an `NCM_<NAME>` environment variable, which is
# how the same notebook is executed headlessly on CPU and on a small GPU (`PRESET="tiny"`).

# %% [markdown]
# ## 0. Knobs

# %%
import os, sys, time, math, json, io, gc, threading, queue, random, warnings, shutil, subprocess
T_START = time.time()
from dataclasses import dataclass

# where are we running? Colab has a ~12 h session limit and /content; RunPod has neither (persistent volume at /workspace)
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
ON_RUNPOD = "RUNPOD_POD_ID" in os.environ or (os.path.isdir("/workspace") and not ON_COLAB)
WORK_DIR = "/workspace" if ON_RUNPOD else ("/content" if os.path.isdir("/content") else os.getcwd())

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")      # flaky links (RunPod): the 10 s default times out on 100 MB parquet files
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

def _knob(name, default):
    """Env override `NCM_<name>` for headless runs; the literal in this cell is the default."""
    v = os.environ.get("NCM_" + name)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    return type(default)(v)

# ---- model + run size ----
PRESET   = _knob("PRESET", "full")     # "full" 1.66B total / 360M active | "tiny" same code at d=512 (any 8 GB GPU / CPU)
TIER     = _knob("TIER", "T1")         # T0 0.02B | T1 2B | T2 20B | T3 50B live codes (see budget cell). T0 = smoke test; set "T1" for the pilot
TOKENS   = _knob("TOKENS", 0.3)        # billions of live codes; 0 = take it from TIER
STEPS    = _knob("STEPS", 0)           # >0 = hard cap on optimizer steps (smoke / CI)
SEED     = _knob("SEED", 0)

# ---- speed ----
DEVICE   = _knob("DEVICE", "auto")     # "auto" | "cuda" | "cpu"
COMPILE  = _knob("COMPILE", True)      # torch.compile the step (per pack shape, static shapes)
COMPILE_MODE = _knob("COMPILE_MODE", "default")   # "max-autotune-no-cudagraphs" for the long run
ATTN     = _knob("ATTN", "levels")     # "levels" (default: one unmasked SDPA call per level, the mask is block-causal over levels; +17% codes/s with flat dispatch on an RTX PRO 6000) | "auto" | "flex" | "sdpa"
SWIGLU   = _knob("SWIGLU", "auto")     # "auto" | "triton" | "torch"
GPU_JPEG = _knob("GPU_JPEG", True)     # decode JPEG on the GPU (nvJPEG) instead of in the workers
MOE_CF   = _knob("MOE_CF", 0.5)        # expert capacity as a fraction of the level's tokens (2 experts x 0.5 = 1 active/token)
MOE_DISPATCH = _knob("MOE_DISPATCH", "flat")     # "flat" (default: index_select / index_add_ on flattened rows, same maths, fewer copies) | "gather" (reference)
RESID    = _knob("RESID", "fp32")      # residual-stream dtype under autocast: "fp32" (default) | "bf16" (measured on an RTX PRO 6000: no speed gain, same loss)
PROFILE  = _knob("PROFILE", False)     # run the profile + A/B cell after the correctness checks (prints which speed knobs to set)
MICRO_TOKENS = _knob("MICRO_TOKENS", 0)   # packed positions per micro-batch; 0 = preset default
BATCH_CODES  = _knob("BATCH_CODES", 0)    # live codes per optimizer step; 0 = preset default

# ---- data: streamed from the Hub; only the encoded codes may be kept (shards) ----
SHARD_DIR  = _knob("SHARD_DIR", os.path.join(WORK_DIR, "ncm_shards"))  # "" = no shards
SHARD_MODE = _knob("SHARD_MODE", "off")      # "off" = stream live (default; the learned tokenizers do not support shards) | "encode" | "record" | "read"
META_ONLY  = _knob("META_ONLY", False) or SHARD_MODE == "encode"   # encode-only runs never build, benchmark or train the model
EVAL_ONLY  = _knob("EVAL_ONLY", False)       # the eval notebook: restore CKPT_DIR/ckpt.pt and run only evaluation (no training, no benchmark)
ACT_CKPT = _knob("ACT_CKPT", -1)     # activation checkpointing: -1 = preset default, 0 = off (faster, needs ~2x activation memory; fine on 96 GB), 1 = on
STREAM_PREFETCH = _knob("STREAM_PREFETCH", 1)   # 1 = Parquet read-ahead + 128 MiB requests when the source allows it
WORKERS  = _knob("WORKERS", 0)         # processes streaming from the Hub; 0 = preset default
PREFETCH = _knob("PREFETCH", 6)        # ready micro-batches held in RAM
RAM_ONLY = _knob("RAM_ONLY", True)     # HF cache dir on /dev/shm, removed at exit
FIT_IMAGES = _knob("FIT_IMAGES", 0)    # images used to fit the frozen codebooks; 0 = preset default
FIT_AUDIO  = _knob("FIT_AUDIO", 0)
VAL_ROWS   = _knob("VAL_ROWS", 0)      # held-out rows per row type; 0 = preset default

# ---- tokenizers, audio decode, caption guidance ----
TOKENIZER  = _knob("TOKENIZER", "feat")  # "feat" = frozen DINOv2 + DAC features, tree-quantised (section 3b; no image/pixel decoder) | "vq" = learned multi-scale residual VQ-VAEs, trained at startup | "kmeans" = the old frozen k-means codecs
TOK_STEPS  = _knob("TOK_STEPS", 0)     # tokenizer training steps; 0 = by tier (T0 600 | T1 4000 | larger 8000)
TOK_IMAGES = _knob("TOK_IMAGES", 0)    # images held in RAM to train the image tokenizer; 0 = by tier (T0 1024 | T1+ 8192)
TOK_AUDIO  = _knob("TOK_AUDIO", 0)     # clips held in RAM to train the audio tokenizer; 0 = by tier (T0 512 | T1+ 2048)
TOK_GATE_DB = _knob("TOK_GATE_DB", 0.5)  # stop before training the model if learned image PSNR is not this many dB above k-means (-99 = never stop)
AUDIO_INV  = _knob("AUDIO_INV", "nnls")  # mel -> spectrum before Griffin-Lim: "nnls" (positive, multiplicative updates) | "pinv" (old, goes negative)
AUDIO_NNLS_ITERS = _knob("AUDIO_NNLS_ITERS", 60)
LEVEL_W    = _knob("LEVEL_W", "slots")  # loss weight per level: "slots" = every position counts the same (a level's weight = its slot count: the old behaviour) | "sqrt" = weight sqrt(slots) | "equal" = every level counts the same
CAP_DROP   = _knob("CAP_DROP", 0.1)    # fraction of image/audio training rows whose caption/transcript is blanked (makes classifier-free guidance possible)
CFG_SCALE  = _knob("CFG_SCALE", 3.0)   # default guidance scale in the generate cells (1 = off; only meaningful if CAP_DROP > 0)

# ---- optimisation ----
LR       = _knob("LR", 0.0)            # 0 = preset default
WARMUP   = _knob("WARMUP", 0)
WD       = _knob("WD", 0.1)
CLIP     = _knob("CLIP", 1.0)

# ---- long-run handling ----
CKPT_DIR = _knob("CKPT_DIR", "")       # e.g. /content/drive/MyDrive/ncm_moe ; "" = no checkpoints
CKPT_MIN = _knob("CKPT_MIN", 30)       # minutes between checkpoints
SESSION_HOURS = _knob("SESSION_HOURS", 11.5 if ON_COLAB else 0.0)   # watchdog: save and stop cleanly before the session limit (0 = off: RunPod has none)
# A Colab disk dies with the session, and `CKPT_DIR=""` means nothing is ever written. The T1 run
# kept ONE rolling checkpoint on ephemeral disk and lost its best audio model (held-out 1.52 at
# 0.73B codes) to a late overfit overwrite. So on Colab, mount Drive and put checkpoints on it.
if ON_COLAB and not CKPT_DIR:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        CKPT_DIR = "/content/drive/MyDrive/ncm_moe"
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"checkpoints: Google Drive mounted -> {CKPT_DIR} (+ <name>_best.pt held-out)")
    except Exception as _e:                                    # noqa: BLE001
        print(f"!! Drive mount failed ({type(_e).__name__}: {_e}). Set NCM_CKPT_DIR=<persistent path>"
              " or this run writes no checkpoints at all.")
EVAL_EVERY_B  = _knob("EVAL_EVERY_B", 0.25)     # evaluate every N billion live codes
LOG_EVERY = _knob("LOG_EVERY", 10)
INSTALL  = _knob("INSTALL", True)

# ---- sources: set a name to 0 to disable it (weights inside a row type are renormalised) ----
SOURCE_WEIGHTS = {}                    # e.g. {"fineweb": 0.5, "libri": 0}; empty = defaults in the data cell

print(f"preset {PRESET} | tier {TIER} | compile {COMPILE} | attn {ATTN} | swiglu {SWIGLU} | cf {MOE_CF}")
print(f"environment: {'Colab' if ON_COLAB else 'RunPod' if ON_RUNPOD else 'other'} | work dir {WORK_DIR} | checkpoints: {CKPT_DIR or 'OFF'}"
      + (" | !! set CKPT_DIR (e.g. /workspace/ncm_moe_ckpt) for any run you want to keep or resume" if ON_RUNPOD and not CKPT_DIR and TIER != "T0" else ""))

# %% [markdown]
# ## 0.1 Runtime: RAM-only HF cache, deps, determinism

# %%
# Streaming does not write dataset shards, but the hub client keeps small metadata and lock files.
# Point them at RAM so nothing survives the process; this is asserted in the no-storage check.
_RAM_HF = None
if RAM_ONLY:
    _base = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    _RAM_HF = os.path.join(_base, f"ncm_hf_{os.getpid()}")
    os.makedirs(_RAM_HF, exist_ok=True)
    for _v in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
        os.environ[_v] = _RAM_HF
    import atexit
    atexit.register(lambda: shutil.rmtree(_RAM_HF, ignore_errors=True))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

if INSTALL:
    _need = []
    for _m, _p in (("datasets", "datasets"), ("soundfile", "soundfile"), ("PIL", "pillow"), ("tqdm", "tqdm")):
        try:
            __import__(_m)
        except ImportError:
            _need.append(_p)
    if _need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need], check=True)

def _secret(name):
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        from google.colab import userdata
        return userdata.get(name) or ""
    except Exception:
        return ""

if _secret("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _secret("HF_TOKEN")      # raises rate limits; unlocks gated sources

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("torch", torch.__version__, "| HF_TOKEN", bool(os.environ.get("HF_TOKEN")), "| HF cache ->", _RAM_HF)
