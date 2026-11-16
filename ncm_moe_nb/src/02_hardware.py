# %% [markdown]
# ## 1. Hardware and kernel probe
#
# Every fast path is probed here and printed, so a silent fallback to a slow kernel shows up as a
# line of text instead of a lower tokens/s three hours later.

# %%
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = DEVICE == "cuda"
dev = torch.device(DEVICE)

# Approximate dense bf16 peaks (TFLOPS) used only to print MFU; unknown cards print TFLOPS only.
_PEAK = {"H100 PCIe": 756, "H100": 989, "H200": 989, "A100": 312, "L4": 121, "4090": 165, "3090": 71, "3070": 40, "T4": 65, "V100": 125,
         "RTX PRO 6000": 500,                                              # Blackwell workstation, dense bf16, approximate (from memory: check the datasheet)
         "MI300X": 1307, "MI325X": 1307, "MI250": 362}                     # AMD numbers are the dense bf16 spec peaks, from memory
IS_ROCM = bool(getattr(torch.version, "hip", None))                         # AMD build of PyTorch: devices still appear as "cuda"
GPU_NAME, VRAM_GB, PEAK_TFLOPS = "cpu", 0.0, 0.0
if IS_CUDA:
    _p = torch.cuda.get_device_properties(0)
    GPU_NAME, VRAM_GB = _p.name, _p.total_memory / 2 ** 30
    PEAK_TFLOPS = next((v for k, v in _PEAK.items() if k in GPU_NAME), 0.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
BF16 = IS_CUDA and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16 else (torch.float16 if IS_CUDA else torch.bfloat16)

def _probe_triton():
    try:
        import triton  # noqa: F401
        return IS_CUDA
    except Exception:
        return False

def _probe_flex():
    if not IS_CUDA or IS_ROCM:                                              # flex-attention is not trusted on ROCm: use ATTN="sdpa" or "levels"
        return False
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask  # noqa: F401
        return True
    except Exception:
        return False

def _probe_nvjpeg():
    if not IS_CUDA or IS_ROCM:                                              # nvJPEG is NVIDIA-only: JPEGs are then decoded on the CPU workers
        return False
    try:
        import torchvision
        from torchvision.io import decode_jpeg  # noqa: F401
        return True
    except Exception:
        return False

def _usable_cpus():
    """CPUs this process may really use: os.cpu_count() reports the HOST (256 on a RunPod H100 pod limited to 24 vCPU)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    try:                                                                    # cgroup v2 quota: "max 100000" or "2400000 100000"
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            n = min(n, max(1, int(q) // int(p)))
    except Exception:
        pass
    try:                                                                    # cgroup v1 quota
        q, p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()), int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            n = min(n, max(1, q // p))
    except Exception:
        pass
    return n

N_CPUS = _usable_cpus()
HAS_TRITON, HAS_FLEX, HAS_NVJPEG = _probe_triton(), _probe_flex(), _probe_nvjpeg()
print(f"device {DEVICE} | {GPU_NAME} | {VRAM_GB:.0f} GiB | bf16 {BF16} | peak ~{PEAK_TFLOPS} TFLOPS | {'ROCm/HIP ' + str(torch.version.hip) if IS_ROCM else 'CUDA'}")
print(f"triton {HAS_TRITON} | flex_attention {HAS_FLEX} | nvJPEG {HAS_NVJPEG} | grouped_mm {hasattr(torch, '_grouped_mm')} "
      f"| cpus {N_CPUS} usable (host reports {os.cpu_count()})")
try:                                                                        # the streaming workers hand batches over through shared memory
    _shm_gb = shutil.disk_usage("/dev/shm").total / 2 ** 30
    print(f"/dev/shm {_shm_gb:.1f} GiB" + ("  !! small: raise the pod's shared memory (RunPod: container settings) or the workers will crash with 'bus error'" if _shm_gb < 4 else ""))
except Exception:
    pass
