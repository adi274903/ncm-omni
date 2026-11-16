# Running on RunPod (H100 SXM)

Nothing here has been run on an H100 yet. Every speed number in the notes comes from one RTX PRO 6000; the sweep below measures the H100.

## Pod

- Template: a recent PyTorch image (torch 2.6 or newer, CUDA 12.x). `pip install -U torch` if the image is older.
- GPU: 1 x H100 SXM (80 GB). The notebook turns activation checkpointing off on cards with 70 GB or more (50 GiB peak measured on the RTX PRO 6000 at micro 32768).
- Volume: at least 100 GB mounted at `/workspace` (a checkpoint is about 20 GB and is written atomically, so allow 2x).
- Shared memory: 4 GiB or more. The notebook prints `/dev/shm` at startup and warns if it is small.
- vCPUs: the data feed needs roughly 12 rows/s per worker and the H100 may need about twice the rows/s the RTX PRO 6000 did. If the log says `data-wait` is high, raise `NCM_WORKERS` up to the vCPU count.

## One-time setup

```bash
cd /workspace && git clone <your repo> ncm && cd ncm/ncm_moe_nb      # or upload the folder
pip install python-dotenv datasets soundfile pillow tqdm nbformat
echo "HF_TOKEN=hf_xxx" > .env                                       # lifts the Hugging Face rate limit
```

## 1. Speed sweep (about 15-25 min per micro-batch size)

```bash
NCM_MICRO_TOKENS=32768 python run_src.py --upto 09 --extra tune_extra.py
NCM_MICRO_TOKENS=49152 python run_src.py --upto 09 --extra tune_extra.py   # may be tight on 80 GB; a failure just reports peak memory
```

Optional extras: `NCM_ATTN=auto` (adds flex attention to the sweep), `NCM_TUNE_RESID=1`, `NCM_TUNE_AUTOTUNE=1` (adds 15-20 min).
The sweep prints a table and a final `use: NCM_ATTN=... NCM_MOE_DISPATCH=...` line, and writes `tune_results_*.json` to `/workspace`.
It uses synthetic batches, so it ignores the data feed. Confirm the winner with a short real run:

```bash
NCM_ATTN=levels NCM_MOE_DISPATCH=flat NCM_TIER=T0 python run_src.py       # read data-wait and codes/s at steps 10 and 20
```

## 2. The real run

```bash
nohup env NCM_TIER=T2 NCM_CKPT_DIR=/workspace/ncm_moe_ckpt <winning knobs> python run_src.py > train.log 2>&1 &
tail -f train.log
```

- Checkpoints go to `CKPT_DIR` every `CKPT_MIN` minutes. Re-running the same command resumes from the checkpoint.
- Do not reuse a `CKPT_DIR` from a smoke test: the run resumes whatever is there.
- `SESSION_HOURS` defaults to 0 (off) outside Colab, so there is no watchdog.
- To use the notebook instead: `python build_nb.py --out NCM_MoE_Omni_RunPod.ipynb` and open it in the pod's JupyterLab. A closed browser tab does not stop the kernel, but the output is lost, so the `nohup` route is safer for long runs.

## Defaults that changed for this

- `ATTN="levels"`, `MOE_DISPATCH="flat"` (+17% on the RTX PRO 6000; untested on H100, so the sweep above decides).
- `ACT_CKPT` off and `MICRO_TOKENS=49152` by VRAM (>= 70 GB / >= 90 GB). The 49152 default does not apply to an 80 GB H100.
- `WORKERS` up to 32 (was 16), capped by the CPU count.
- H100 PCIe peak 756 TFLOPS, H100 SXM 989, for the MFU line.
