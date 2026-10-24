#!/usr/bin/env python3
"""Build NCM_Omni_Colab_v3.ipynb from the .py files in this repo.

The notebook carries a verbatim copy of all 13 shipped files (core.py, run_omni.py, omni/*.py),
because a Colab notebook that `import`s from a clone runs whatever the clone happened to be - and
that is exactly the failure mode the snapshot removes. A copy that is not regenerated is a copy
that lies, so this script is the only writer:

    python tools/build_notebook.py            # regenerate the notebook
    python tools/build_notebook.py --check    # exit 1 if any payload cell drifted from the file

Every file's sha256 goes into the notebook's EXPECTED table, and the notebook verifies it at
runtime, so a hand-edited or truncated cell stops the run instead of training on mystery code.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "NCM_Omni_Colab_v3.ipynb")

# Order is the order the files appear in the notebook. Only readability depends on it: every file
# is on disk before the first import. core.py is first because omni/model.py imports it by bare
# name (`from core import Block, ...`), which is the one non-obvious dependency in the tree.
MANIFEST = [
    "core.py",
    "run_omni.py",
    "omni/__init__.py",
    "omni/schedule.py",
    "omni/codecs.py",
    "omni/producer.py",
    "omni/livestream.py",
    "omni/data.py",
    "omni/model.py",
    "omni/moe.py",
    "omni/preflight.py",
    "omni/train.py",
    "omni/eval.py",
]

STAGE_REL = ".ncm_stage"          # payload cells write here, never into PKG


def sha12(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def expected_table():
    return {rel: sha12(os.path.join(ROOT, rel)) for rel in MANIFEST}


def read_src(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def md(text):
    return dict(cell_type="markdown", metadata={}, source=_lines(text))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=_lines(text))


def _lines(text):
    """nbformat stores source as one entry per line, trailing newlines included."""
    rows = text.splitlines(keepends=True)
    return rows or [""]


def payload_cell(rel, src):
    """`%%writefile` is a verbatim write: the cell body IS the file, with no escaping to get wrong.

    It writes to the staging dir so that git mode cannot be clobbered by the snapshot after the
    clone - the two sources are reconciled in the verify cell, not silently overwritten.
    """
    return code(f"%%writefile {STAGE_REL}/{rel}\n{src}")


def n_lines(rels):
    return sum(len(read_src(r).splitlines()) for r in rels)


def build():
    exp = expected_table()
    manifest_rows = "\n".join(
        f"| `{rel}` | {len(read_src(rel).splitlines())} | `{exp[rel]}` |" for rel in MANIFEST)
    exp_lit = json.dumps(exp, indent=4).replace("\n", "\n")

    cells = []

    cells.append(md(f"""# NCM Omni - Colab, self-contained, direct feed

One arm, one loss, one path from the Hub to weights. **The code ships inside this notebook**: {len(MANIFEST)}
files, {n_lines(MANIFEST):,} lines, materialised to `{STAGE_REL}` and `PKG` by the cells below, with a
sha256 check before anything imports. Nothing is `pip install`ed from git and nothing is imported
from a clone, so a session cannot silently run yesterday's model.

Two sources, one switch (`SRC` in the knobs cell):

| `SRC` | what happens | when |
|---|---|---|
| `"snapshot"` | the cells below write {len(MANIFEST)} files and the run uses exactly that code | default; reproducible; offline |
| `"git"` | `PKG` is deleted and re-cloned at `PIN`, then sha-checked against the snapshot | you pushed a fix and want it without restarting the kernel |

Both paths end in the same place: `sys.modules` purged for `omni*`, `core`, `run_omni`; bytecode
caches invalidated; `PKG` on `sys.path`; the sha table compared and printed. The purge half matters
as much as the pull: a re-clone in a live kernel keeps executing the old module objects until the
kernel restarts, and the only symptom is "my fix did nothing".

Shipped files:

| file | lines | sha256[:12] |
|---|---|---|
{manifest_rows}

Not shipped: `tests/`, `_probe_sources.py`, `run/` (artefacts), `.env` (secrets). Regenerate this
notebook with `python tools/build_notebook.py`; `--check` fails if a payload cell drifted."""))

    cells.append(md("## 0. Knobs - one cell drives the whole run"))

    cells.append(code(f'''# ---- where the code comes from ----
# "snapshot": the {len(MANIFEST)} cells further down are the code. No network, no clone.
# "git":      PKG is deleted and re-cloned at PIN. The purge-and-re-pull path.
SRC = "snapshot"

PIN = "main"        # git mode: branch, tag or commit sha. Only a sha is reproducible.
GIT_URL = "https://github.com/adi274903/ncm-omni-v2.git"
GH_PAT = ""         # private repo only; filled from a Colab secret in the next cell

PKG = "/content/ncm_omni"     # where the package lands - and the cwd for every artefact below
STAGE = PKG.rsplit("/", 1)[0] + "/{STAGE_REL}"   # payload staging dir, deleted after the copy

# ---- model: d=1536, L=16, 3 experts x 360.0M = 1.178B total, 462M active ----
PRESET = "moex"     # "sm" | "md" | "lg" | "moex"
MOE = 3             # one expert per modality, routing forced by packed position
MOE_FFN = 4880      # expert SwiGLU width -> 360.0M per expert, h/d = 3.18
                    # 0 = fall back to mlp_ratio 4.0 (2048 at d=768), which is not 360M

# ---- what the run is allowed to spend ----
TOKENS = 0.002      # BILLIONS of live code predictions. The budget; STEPS is derived from it.
BATCH = 32          # 0 = derive from VRAM
TFLOPS = 0          # 0 = auto: 25 on this box's 3070 (20-30 measured), 125 on A100, 15 on T4
LEN = 256           # base_len for text and image. Longer samples = fewer rows for the same token
                    # count, which matters because the Hub feed, not the GPU, sets the pace.
EVAL_EVERY = 50
LOG_EVERY = 10
HOLD = 64           # rows pulled off the front of the feed and held out (direct-feed path only)
PRE_STEPS = 60      # preflight steps. The gate is cheap at 1.18B and catches a dead head early.
PRE_BATCH = 4
USE_SHARDS = True   # produce first, train from npz: a ~2 samples/s feed cannot keep 1.18B busy
INSTALL = True      # pip-install the runtime deps
SMOKE = 0           # >0 = timing-only run of that many steps
ARM = "strict"

# sha256[:12] of every file this notebook carries. Generated by tools/build_notebook.py - if a
# payload cell above was edited by hand, this table is stale and the verify cell will say so.
EXPECTED = {exp_lit}

print(f"{{SRC}} mode | {{len(EXPECTED)}} files | arm {{ARM}} | batch {{BATCH}} | budget {{TOKENS}}B live codes | PKG {{PKG}}")'''))

    cells.append(md("""## 0.1 Runtime deps

`torch` is on the Colab image already and is deliberately not pinned here: a plain
`pip install -r requirements.txt` can replace a CUDA 13 wheel with a CPU one, and the failure shows
up as a slow run, not an error."""))

    cells.append(code('''import os
import subprocess
import sys

RUNTIME = "datasets huggingface_hub soundfile av pillow tqdm tensorboard".split()
if INSTALL:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *RUNTIME], check=True)

# Must be set before the first CUDA allocation to matter. At 1.198B with fp32 params + AdamW there
# is no slack: without expandable segments a freed 19 GiB model cannot be re-used for the next one,
# and the OOM blames a 30 MiB tensor with 38 GiB "in use".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("alloc conf:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("deps:", " ".join(RUNTIME), "| installed" if INSTALL else "| skipped")'''))

    cells.append(md("""## 1. Credentials

Colab secrets first (they survive a restart), then the process environment. `HF_TOKEN` only raises
rate limits and unlocks gated sources; the git path needs a PAT only for a private clone."""))

    cells.append(code('''import os


def _secret(name):
    """-> value from the env, else from Colab's secret store, else ""."""
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        from google.colab import userdata
        return userdata.get(name) or ""
    except Exception:
        return ""


HF_TOKEN = _secret("HF_TOKEN")
GH_PAT = GH_PAT or _secret("GITHUB_PAT") or _secret("GIT_PAT") or _secret("GITHUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN        # datasets / huggingface_hub read this directly
print({"HF_TOKEN": bool(HF_TOKEN), "GITHUB PAT": bool(GH_PAT), "SRC": SRC})'''))

    cells.append(md("""## 2. Purge, then get the code

The order is the point: forget the modules, delete the directory, *then* put code back. Deleting
files while `omni.model` sits in `sys.modules` is the re-pull that appears to do nothing.

`git` mode fetches `PIN` into an empty repo rather than cloning a branch, so a commit sha works
exactly like a branch name."""))

    cells.append(code('''import importlib
import os
import shutil
import subprocess
import sys


def purge_modules():
    """Drop every module this notebook is about to redefine, and the bytecode caches.

    Files on disk are not enough: Python caches the module object, so a re-pull in a live kernel
    keeps executing the old code and the only symptom is "my fix did nothing".
    """
    mine = [m for m in list(sys.modules)
            if m == "omni" or m.startswith("omni.") or m in ("core", "run_omni")]
    for m in mine:
        del sys.modules[m]
    importlib.invalidate_caches()
    return sorted(mine)


PARENT = os.path.dirname(PKG.rstrip("/")) or "/"
os.makedirs(PARENT, exist_ok=True)
os.chdir(PARENT)                     # never stand inside the directory about to be deleted

dropped = purge_modules()
shutil.rmtree(STAGE, ignore_errors=True)
shutil.rmtree(PKG, ignore_errors=True)
os.makedirs(os.path.join(STAGE, "omni"))
print("purged modules:", dropped or "none")
print(f"staging {STAGE} <- the {len(EXPECTED)} payload cells below")

if SRC == "git":
    url = GIT_URL.replace("https://", f"https://x-access-token:{GH_PAT}@") if GH_PAT else GIT_URL
    os.makedirs(PKG)
    steps = [["git", "init", "-q"],
             ["git", "remote", "add", "origin", url],
             ["git", "fetch", "-q", "--depth", "1", "origin", PIN],
             ["git", "checkout", "-q", "-f", "FETCH_HEAD"]]
    for argv in steps:
        print("+", argv[0], argv[1], "..." if argv[-1] == url else argv[-1])
        subprocess.run(argv, cwd=PKG, check=True, capture_output=True, text=True)
    head = subprocess.run(["git", "-C", PKG, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print("git: checked out", PIN, "->", head)'''))

    cells.append(md(f"""## 2.1 The {len(MANIFEST)} files

Each cell is a verbatim `%%writefile` of one file into `{STAGE_REL}/`. Nothing here is edited inside the
notebook - edit the `.py` in the repo and run `python tools/build_notebook.py`, or the sha table and
the body will disagree and the verify cell will refuse to continue.

They write to a staging dir, never straight into `PKG`: in `git` mode the clone must win, and a
payload cell that overwrote it afterwards would make `SRC="git"` a lie."""))

    for rel in MANIFEST:
        cells.append(payload_cell(rel, read_src(rel)))

    cells.append(md("""## 3. Verify, then import

`git` mode compares the clone against the snapshot and *warns* on a difference - a pinned repo that
is behind the notebook is a fact worth seeing, not an error. `snapshot` mode *fails* on a
difference, because there the payload cell is the only copy of the code."""))

    cells.append(code('''import hashlib
import os
import shutil
import sys


def _sha12(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


if SRC == "git":
    print("source: git", PIN, "->", PKG, "| the clone is authoritative")
else:
    shutil.copytree(STAGE, PKG, dirs_exist_ok=True)
    print("source: snapshot ->", PKG)
shutil.rmtree(STAGE, ignore_errors=True)          # only after the copy landed

got = {}
for rel, want in EXPECTED.items():
    path = os.path.join(PKG, rel)
    got[rel] = _sha12(path) if os.path.exists(path) else None

missing = [r for r, h in got.items() if h is None]
differ = [r for r, h in got.items() if h is not None and h != EXPECTED[r]]
assert not missing, f"missing after materialising: {missing}"

if differ:
    print(f"\\n!! {len(differ)} file(s) differ from this notebook's snapshot (sha256[:12]):")
    for rel in differ:
        print(f"   {rel:20s} notebook {EXPECTED[rel]}  on disk {got[rel]}")
    if SRC == "snapshot":
        raise AssertionError("payload != sha table: regenerate with python tools/build_notebook.py")
    print("   (git mode: the clone wins - the notebook is behind the pushed code)")
else:
    print(f"sha256 OK | {len(EXPECTED)} files | nothing truncated, nothing stale")

if PKG not in sys.path:
    sys.path.insert(0, PKG)
os.chdir(PKG)              # run/, checkpoints and tensorboard all land under PKG

import omni                                    # noqa: E402  (TF poisoning shim, then the package)
import core                                    # noqa: E402  (omni/model.py imports this by name)
from omni.schedule import OmniSchedule         # noqa: E402

assert os.path.dirname(os.path.abspath(omni.__file__)) == os.path.join(PKG, "omni"), \\
    "omni resolved somewhere else - a stale install is shadowing PKG"
print("omni from", os.path.dirname(omni.__file__))
print("core from", core.__file__)'''))

    cells.append(md("""## 4. Hardware

Fail here, not 40 minutes into a feed. `moex` is 1.178B parameters with 462M active per token, and at
`LEN=1024` one sample packs 7,056 positions - so the batch that fits is set by VRAM, not by taste."""))

    cells.append(code('''import torch
from omni.train import fast_math

assert torch.cuda.is_available(), "Runtime -> Change runtime type -> GPU"
p = torch.cuda.get_device_properties(0)
vram = p.total_memory / 2 ** 30
print(f"{p.name} | sm {p.major}.{p.minor} | {vram:.0f} GiB")
print("fast math:", fast_math())

if not BATCH:
    BATCH = 8 if vram < 20 else (16 if vram < 40 else 32)
print(f"batch {BATCH} | 1.178B params + AdamW is ~14 GiB before a single activation")
if vram < 20:
    print("!! 1.178B with AdamW will not sit comfortably in this card: lower BATCH to 8, or free it")'''))

    cells.append(md("""## 5. Config, built in memory

`build_parser` + `apply_overrides` + `train_cfg` are the same three functions the CLI uses, so the
notebook cannot drift from `run_omni.py train`. `ARM = "strict"` is the only arm: every code of a
level in one forward, 6 forwards for 6 levels instead of 504."""))

    cells.append(code('''import os

import torch

import run_omni as R
from omni.producer import GROUPS
from omni.schedule import OmniSchedule
from omni.train import build as build_net
from omni.train import flop_params

args = R.build_parser().parse_args(
    ["train", "--preset", PRESET, "--base-len", str(LEN), "--batch", str(BATCH),
     "--moe", str(MOE), "--moe-ffn", str(MOE_FFN), "--log-every", str(LOG_EVERY)])
rt = R.apply_overrides(R.default_cfg(), args)

# ---- the token budget, resolved before anything is spent ----
# STEPS is a consequence of TOKENS, not the other way round. "Is it learning" is a question about
# tokens seen; a step count means nothing without the width and packed length that produced it.
sched_pre = OmniSchedule(base_len=rt["sched"]["base_len"], K=rt["sched"]["K"], R=rt["sched"]["R"],
                         null_code=rt["sched"].get("null_code", 0),
                         modalities=tuple(rt["sched"]["modalities"]))
mix = rt["produce"]["mix"]
w = sum(mix.values())
LIVE = sum(mix[k] / w * sched_pre.flat_len(GROUPS[k]) for k in mix)   # live codes per sample
T = sched_pre.total_len
STEPS = max(1, round(TOKENS * 1e9 / (BATCH * LIVE)))

cfg = R.train_cfg(rt, ARM, out_dir=os.path.join(PKG, "run", "direct"), steps=STEPS)
cfg["eval_every"] = EVAL_EVERY
cfg["log_every"] = LOG_EVERY
cfg["eval_files"] = {}          # no npz: validation is the holdout
cfg["hard_exit"] = False        # a notebook cell has to return; run() never needs os._exit
cfg["preflight"] = dict(steps=PRE_STEPS, batch=PRE_BATCH, mask_mode=ARM)

with torch.device("meta"):      # count the parameters without allocating 1.178B of them
    net_pre, _ = build_net(cfg)
ACTIVE = flop_params(net_pre)

# Effective bf16, not peak. This box's 3070 measures 20-30 TFLOPS end to end; an A100 lands near
# 125 at ~40% MFU, a T4 nearer 15. Guessing high here is how a "5 minute" run becomes an hour.
if TFLOPS:
    tflops_eff = float(TFLOPS) * 1e12
elif not torch.cuda.is_available():
    tflops_eff = 5e10                                  # CPU: smoke only, this is not a benchmark
else:
    _nm = torch.cuda.get_device_name(0).upper()
    _gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if "A100" in _nm or "H100" in _nm:
        tflops_eff = 125e12
    elif _gb < 20:
        tflops_eff = 25e12                             # consumer card: 20-30 measured
    else:
        tflops_eff = 60e12
est = 6 * ACTIVE * STEPS * BATCH * T / tflops_eff / 60
feed = STEPS * BATCH / 2 / 60                           # Hub feed at ~2 samples/s
print(f"model  {PRESET}: {MOE} experts x {MOE_FFN} wide | {ACTIVE/1e6:.0f}M active/token")
print(f"length {T} positions/sample | {LIVE:.0f} live codes/sample ({100*LIVE/T:.0f}% of slots)")
print(f"budget {TOKENS:.3f}B live codes = {STEPS:,} steps x {BATCH} = {STEPS*BATCH:,} samples")
print(f"eta    ~{est:.0f} min at {tflops_eff/1e12:.0f} effective TFLOPS | "
      f"~{feed:.0f} min if the Hub feed sets the pace")
print(f"mask   {cfg['model']['mask_mode']} | data {'shards' if USE_SHARDS else 'direct feed'}")
print("out", cfg["out_dir"])'''))

    cells.append(md("""## 6. TensorBoard

Started before the first step so step 0 is visible. The path is `PKG/run/direct/tb` - edit it if you
moved `PKG`."""))

    cells.append(code('''%load_ext tensorboard
%tensorboard --logdir /content/ncm_omni/run/direct/tb'''))

    cells.append(md(f"""## 7. The direct feed

`iter_rows` pulls from the Hub, decodes with the frozen codecs and yields encoded rows in-process.
The first `HOLD` rows become the in-memory validation set and are consumed from the same iterator,
so nothing is written to disk and nothing can go stale. Stable within a run; it is not an external
test set."""))

    cells.append(code('''if USE_SHARDS:
    hold, feed = None, None                  # rows come off npz; nothing is pulled here
    print("shard mode: the feed cell is skipped")
else:
    from omni.livestream import iter_rows

    live = dict(rt["produce"])
    live.update(sched=rt["sched"], vocab=rt["vocab"], batch_docs=8)

    raw = iter_rows(live, verbose=True)      # generator: nothing is fetched until it is asked
    import time as _t

    # A silent first pull is indistinguishable from a hang, so report progress. And make the holdout
    # STRATIFIED: the first HOLD rows off the live feed are not a sample of the mix - at startup the
    # cheapest streams (text) dominate - so a first-N holdout can be one modality only, and eval then
    # reports 0.0 accuracy for the others, which reads as a broken model instead of an untested one.
    QUOTA = {"text": 24, "image": 24, "audio": 16}
    _t0, hold, got, pulled = _t.time(), [], {k: 0 for k in QUOTA}, 0
    print(f"opening {len(live['mix'])} streams, pulling a stratified holdout {QUOTA}...", flush=True)
    # Cap on TIME, not row count. Rows are cheap once a pump is up (15/s measured) but the 19 streams
    # take ~30 s to all start, and the image/audio pumps come up last: a row-count cap expires inside
    # that warm-up and the holdout ends up text-only, which then reads as 0.0 accuracy downstream.
    while _t.time() - _t0 < 180:
        r = next(raw)
        pulled += 1
        # Credit the row to the scarcest modality it satisfies, so a text+image row helps image.
        cands = [m for m, ok in r["present"].items() if ok and m in QUOTA and got[m] < QUOTA[m]]
        if cands:
            g = max(cands, key=lambda m: QUOTA[m] - got[m])
            hold.append(r)
            got[g] += 1
        if pulled % 32 == 0:
            print(f"  {pulled} rows, {_t.time() - _t0:.0f}s -> holdout {got}", flush=True)
        if all(got[k] >= QUOTA[k] for k in QUOTA):
            break
    feed = raw
    print(f"holdout {len(hold)} rows {got} in {_t.time() - _t0:.0f}s ({pulled} pulled)")
    missing = [k for k in QUOTA if got[k] < QUOTA[k]]
    if missing:
        print(f"!! never saw enough of {missing} in 180s: eval will skip those modalities. "
              f"Their streams are slow to start or throttled (set HF_TOKEN).")'''))

    cells.append(md("""## 7b. Produce, then train off the shards

`USE_SHARDS = True` builds the corpus once, then the trainer reads npz. At `LEN=256` a sample carries
~546 live codes, so the smoke budget is a few thousand samples, while the direct feed yields ~2
samples/s - an order of magnitude slower than this GPU can consume. Producing also exercises the
encoder, the codec ladder and the shard writer, which a direct-feed run never touches.

The producer writes its own output to `run/producer*.log`, so this cell polls the shard directory and
prints a progress line every 15 s instead of blocking silently."""))

    cells.append(code('''import glob
import json
import subprocess
import sys
import time

if USE_SHARDS:
    n_samples = max(STEPS * BATCH, 512)
    workers = 4 if n_samples >= 512 else 1       # one Hub stream per worker, so 4 is ~4x the rows
    cmd = [sys.executable, os.path.join(PKG, "run_omni.py"), "produce", "--foreground",
           "--budget", str(n_samples), "--base-len", str(LEN), "--shuffle-buf", "512",
           "--workers", str(workers)]
    print(f"$ python run_omni.py produce --budget {n_samples:,} --base-len {LEN} --workers {workers}")

    # The producer redirects its own stdout to run/producer*.log, so subprocess.run() sits completely
    # silent for the whole corpus - and half an hour of nothing looks exactly like a hang. Poll the
    # progress json the producer already writes: it carries the per-stage split, which is the only
    # thing that distinguishes a run waiting on the Hub from one waiting on the codec.
    def _tail(n=1):
        logs = sorted(glob.glob(os.path.join(PKG, "run", "producer*.log")),
                      key=os.path.getmtime) if os.path.isdir(PKG) else []
        if not logs:
            return ""
        lines = [l.strip() for l in open(logs[-1], errors="replace").read().splitlines() if l.strip()]
        return " | ".join(lines[-n:])[:110] if lines else ""

    def _prog():
        ps = glob.glob(os.path.join(cfg["shard_dir"], "progress*.json"))
        if not ps:
            return ""
        samples = budget = 0
        pct = {}
        for f in ps:
            try:
                d = json.load(open(f))
            except (OSError, ValueError):
                continue
            samples += int(d.get("samples", 0))
            budget += int(d.get("budget", 0))
            for k, v in (d.get("stage_pct") or {}).items():
                pct[k] = pct.get(k, 0.0) + float(v) / max(1, len(ps))
        s = f"{samples:,}/{budget:,} ({100 * samples / max(1, budget):.0f}%)"
        if pct:
            s += (f" | pull {pct.get('pull', 0):.0f}% enc {pct.get('enc', 0):.0f}%"
                  f" write {pct.get('write', 0):.0f}%")
        return s

    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=PKG)
    last_n, last_t, warned = -1, time.time(), False
    while p.poll() is None:
        time.sleep(15)
        d = cfg["shard_dir"]
        got = len(glob.glob(os.path.join(d, "shard_*.npz"))) if os.path.isdir(d) else 0
        line = _prog()
        print(f"  {time.time() - t0:5.0f}s | {got:3d} shards | {line} | {_tail()}", flush=True)
        # A Hub retry loop looks identical to a hang from the outside: the sample count freezes and
        # the only clue is one repeated "Retrying in Ns" line. Say so once, with the cause.
        n = int(line.split("/")[0].replace(",", "")) if line else 0
        if n > last_n:
            last_n, last_t, warned = n, time.time(), False
        elif time.time() - last_t > 90 and not warned:
            warned = True
            print(f"  !! no new samples for {time.time() - last_t:.0f}s. If the tail says 'Retrying in "
                  f"Ns', the Hub is throttling this run - set HF_TOKEN in cell 6, or drop --workers.",
                  flush=True)
    if p.returncode != 0:
        # `run_omni.py produce` returns 1 if ANY worker returns non-zero, and the reason lives in
        # that worker's own log - not in the aggregate one. Print all of them, so the traceback is
        # in the cell instead of one grep away.
        print(f"producer exited {p.returncode} after {time.time() - t0:.0f}s - worker log tails:")
        for f in sorted(glob.glob(os.path.join(PKG, "run", "producer*.log"))):
            lines = [l.rstrip() for l in open(f, errors="replace").read().splitlines() if l.strip()]
            print(f"  == {os.path.basename(f)} ({len(lines)} lines)")
            for l in lines[-14:]:
                print("    ", l[:150])
        raise RuntimeError(f"produce failed after {time.time() - t0:.0f}s; see the tails above")

    shards = sorted(glob.glob(os.path.join(cfg["shard_dir"], "shard_*.npz")))
    evals = {os.path.basename(f)[5:-4]: f
             for f in glob.glob(os.path.join(cfg["shard_dir"], "eval_*.npz"))}
    assert len(shards) >= 1, f"no shard_*.npz in {cfg['shard_dir']} - {_tail(15) or 'no log'}"
    for cand in ("producer.log", "producer_w0.log"):
        if os.path.exists(os.path.join(PKG, "run", cand)):
            cfg["producer_log"] = os.path.join(PKG, "run", cand)
            break
    cfg["eval_files"] = evals      # the producer's held-out rows; eval_every reads these, not HOLD
    print(f"shards {len(shards)} | eval sets {sorted(evals)} in {cfg['shard_dir']} "
          f"| {time.time() - t0:.0f}s")
else:
    print("direct-feed mode: no producer run")'''))

    cells.append(md("""## 8. Preflight - the gate, in-process

The synthetic copy-relation stream: every level `k >= 1` must drop by at least 1.5 nats and land
under `0.5 * ln(V)` within `PRE_STEPS`. Catches a broken head or dead routing in a minute instead of
300 wasted GPU-hours. Asserts, because training on a model that cannot learn is the expensive
outcome, not the failed assert."""))

    cells.append(code('''from omni.preflight import run_preflight

res = run_preflight(cfg)
assert res["passed"], f"preflight failed on {res.get('failing')} - head/routing does not learn"
print("PREFLIGHT PASSED")'''))

    cells.append(md("""## 9. Train - one arm, off the feed

`run(cfg, feed=..., eval_rows=...)` bypasses `ShardStream` entirely, so the kernel survives it (no
`os._exit` on this path). Step 0 logs immediately: that line is the proof the real data path works."""))

    cells.append(code('''cfg["preflight"]["enabled"] = False    # already gated above; do not pay twice
from omni.train import run

if USE_SHARDS:
    run(cfg)                              # ShardStream. hard_exit is off, so the cell returns.
else:
    run(cfg, smoke=SMOKE or None, feed=feed, eval_rows=hold)'''))

    cells.append(md("## 10. What came out"))

    cells.append(code('''import json
import math
import os
import statistics

path = os.path.join(cfg["out_dir"], "metrics.jsonl")
rows = [json.loads(l) for l in open(path) if l.strip()]
# metrics.jsonl is opened in append mode, so it also holds every earlier run in this out_dir - and
# an old record's loss has nothing to do with this one. A new run restarts at step 0, so the last
# descending step among the training records marks where this run begins.
start, prev = 0, -1
for i, r in enumerate(rows):
    if r.get("val"):
        continue
    s = r.get("step", 0)
    if s < prev:
        start = i
    prev = s
if start:
    print(f"ignoring {start} records from earlier runs in this file")
rows = rows[start:]
train = [r for r in rows if not r.get("val")]
val = [r for r in rows if r.get("val")]
assert train, f"no training rows in {path} - the run never reached a logged step"

first, last = train[0], train[-1]
print(f"steps {len(train)} | loss {first['loss']:.4f} -> {last['loss']:.4f}"
      f" | {last.get('pos_per_s', 0):,.0f} pos/s | {last.get('tflops', 0)} TFLOPS")

# ---------------------------------------------------------------- is it learning?
# Six assertions on metrics.jsonl alone. Per-(modality, level) CE is in the record, so this needs no
# TensorBoard reader: one expert per modality is only worth it if the three learn separately.
fails = []


def need(ok, why):
    if not ok:
        fails.append(why)


need(all(math.isfinite(r["loss"]) for r in train), "non-finite loss in the record")
n = max(1, len(train) // 5)
early = statistics.median(r["loss"] for r in train[:n])
late = statistics.median(r["loss"] for r in train[-n:])
drop = 100 * (1 - late / early) if early else 0.0
print(f"loss  median {early:.3f} -> {late:.3f}  ({drop:+.1f}%, ln(257) at init = {math.log(257):.3f})")
need(late <= 0.83 * early, f"loss moved only {drop:.1f}%, want >= 17%")

per = {}
for r in train:
    for key, v in (r.get("ce") or {}).items():
        per.setdefault(key, []).append(v)
moved, flat = [], []
for mod in ("text", "image", "audio"):
    ks = sorted(k for k in per if k.startswith(mod + "_"))
    if not ks:
        continue
    m = min(len(per[k]) for k in ks)
    series = [sum(per[k][i] for k in ks) / len(ks) for i in range(m)]
    d = 100 * (1 - series[-1] / series[0]) if series[0] else 0.0
    print(f"  {mod:5s} CE {series[0]:6.3f} -> {series[-1]:6.3f}  ({d:+5.1f}%)")
    (moved if d >= 10 else flat).append(mod)
need(len(moved) >= 2, f"{len(moved)}/3 modalities dropped >=10% ({flat} flat)")

best = {}
for r in val:
    for key, v in r["val"].items():
        if v.get("acc", 0.0) > best.get(key, 0.0):
            best[key] = v["acc"]
mini = max(0.05, 12 * 1 / 257)              # chance is 1/257 = 0.39%
won = {m for k, a in best.items() if a >= mini for m in [k.split("_")[0]]}
if best:
    print("val   best acc per modality:",
          {m: round(max([a for k, a in best.items() if k.startswith(m + '_')] or [0.0]), 3)
           for m in ("text", "image", "audio")})
    need(len(won & set(moved)) >= 2, f"val accuracy near chance for 2+ modalities ({sorted(won)})")
else:
    print("val   no eval sets - accuracy check skipped (shard mode reads eval_*.npz from produce)")

need(all(math.isfinite(r.get("grad_norm", 1.0)) for r in train), "non-finite grad norm")
need(last["pos_per_s"] > 0, "throughput is zero")
need(len(train) >= 3, "fewer than 3 logged steps: nothing to conclude")

print()
if fails:
    print("VERDICT:", "NOT LEARNING" if late > 0.95 * early else "PARTIAL")
    for f in fails:
        print("  -", f)
    print("  Do not scale this up. Fix the gate first, then turn on one modality at a time.")
else:
    # Report what actually ran, not what was planned: the shard stream ends when the corpus does,
    # so `STEPS x BATCH` can overstate the run by an order of magnitude (47 of 229 steps here).
    seen = int(last.get("samples") or 0) or STEPS * BATCH
    print(f"VERDICT: PASS | loss {drop:.1f}% down | {len(moved)}/3 modalities moving | "
          f"{seen:,} samples = {seen * LIVE / 1e6:.2f}M live codes")
    if seen < STEPS * BATCH:
        print(f"  note: ran {len(train)} logged steps of a planned {STEPS} - the corpus ran out. "
              f"{seen:,} of {STEPS * BATCH:,} samples.")
    if not best:
        print("  note: no held-out validation in this run (produce never wrote eval_*.npz), "
              "so this is training loss only.")

print("tb:", os.path.join(cfg["out_dir"], "tb"))'''))


    cells.append(md("""## 10b. Where the time went

`metrics.jsonl` carries a per-step stage split, taken with the GPU synchronised, so it measures work
and not kernel launch: `t_fetch` is the feed, `t_prep` is collate + host-to-device, `t_fwd`/`t_bwd`/
`t_opt` are the model. On a feed-bound run `data` sits near 90% and no amount of GPU tuning helps."""))

    cells.append(code('''import json
import os
import statistics

rows = [json.loads(l) for l in open(os.path.join(cfg["out_dir"], "metrics.jsonl")) if l.strip()]
train = [r for r in rows if not r.get("val") and "t_fetch" in r]
assert train, "no stage timings in metrics.jsonl - this run predates the trace change"

keys = ("t_fetch", "t_prep", "t_fwd", "t_bwd", "t_opt")
med = {k: statistics.median(r.get(k, 0.0) for r in train) for k in keys}
tot = max(1e-9, sum(med.values()))
print(f"{len(train)} logged steps | median step {tot:.3f}s")
print("  " + " ".join(f"{k.replace('t_', ''):5s} {med[k]:6.3f}s {100 * med[k] / tot:5.1f}%"
                      for k in keys))
data = 100 * (med["t_fetch"] + med["t_prep"]) / tot
print(f"\\ndata {data:.0f}% | gpu {100 - data:.0f}%")
if data > 60:
    print("  feed-bound: the GPU idles most of the step. Produce ahead of training, or add workers.")
else:
    print("  compute-bound: the feed is keeping up.")'''))
    cells.append(md(f"""## Appendix - re-running after a fix

| you changed | do this |
|---|---|
| a `.py` in the repo | `python tools/build_notebook.py`, upload the notebook, **Run all** |
| pushed a commit, want it now | set `SRC = "git"` and `PIN = "<sha>"`, re-run cells 2-3 only |
| nothing, but the kernel is confused | re-run the purge cell, then the verify cell |

Why the purge cell is not optional in git mode: `sys.modules` holds the module *objects*. Re-pulling
files under a live kernel leaves `omni.model.OmniNCM` pointing at the old code, so the run continues
with the version you were trying to replace. Cell 2 deletes `omni*`, `core` and `run_omni` from
`sys.modules` and calls `importlib.invalidate_caches()` before anything is copied; the verify cell
then imports and asserts that `omni` resolved to `PKG/omni` and not to a site-packages copy.

`tools/build_notebook.py --check` compares every payload cell against the file on disk and exits
non-zero on drift, which is the sane thing to put in CI."""))

    nb = dict(cells=cells,
              metadata=dict(language_info=dict(name="python", version="3.11.0"),
                            colab=dict(provenance=[], toc_visible=True),
                            kernelspec=dict(display_name="Python 3", name="python3")),
              nbformat=4, nbformat_minor=0)
    return nb


def check(nb_path):
    """Exit non-zero if any payload cell or the sha table disagrees with the repo."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    seen, bad = {}, []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not src.startswith("%%writefile "):
            continue
        head, _, body = src.partition("\n")
        rel = head.split("/", 1)[1]
        seen[rel] = body

    for rel in MANIFEST:
        want = read_src(rel)
        if rel not in seen:
            bad.append(f"{rel}: no payload cell")
        elif seen[rel] != want:
            bad.append(f"{rel}: payload cell differs from the file on disk")
    for rel in sorted(set(seen) - set(MANIFEST)):
        bad.append(f"{rel}: payload cell for a file that is not in MANIFEST")

    cfg_src = next(("".join(c["source"]) for c in nb["cells"]
                    if "EXPECTED = {" in "".join(c["source"])), "")
    start = cfg_src.find("EXPECTED = ") + len("EXPECTED = ")
    end = cfg_src.find("\n}", start) + 2
    try:
        table = json.loads(cfg_src[start:end])
    except json.JSONDecodeError as e:
        table = {}
        bad.append(f"EXPECTED table is not parsable JSON: {e}")
    for rel, want in expected_table().items():
        if table.get(rel) != want:
            bad.append(f"{rel}: sha table {table.get(rel)} != {want}")

    if bad:
        print("DRIFT:")
        for b in bad:
            print(" ", b)
        return 1
    print(f"OK: {len(MANIFEST)} payload cells and the sha table match the repo")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify an existing notebook instead of writing a new one")
    a = ap.parse_args(argv)

    if a.check:
        return check(a.out)

    nb = build()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {a.out}")
    print(f"  {len(nb['cells'])} cells ({n_code} code) | {len(MANIFEST)} files | "
          f"{n_lines(MANIFEST):,} lines of shipped source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
