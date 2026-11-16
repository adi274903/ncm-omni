# %% [markdown]
# ## 2. Presets, the level ladder and the pack layout
#
# **The ladder.** `K` levels, ratio `R`. Level *k* of a modality with finest length `n` has
# `n / R^(K-1-k)` slots, so level 0 is the whole sample in a handful of codes and level `K-1` is the
# finest detail. Images use a 2D grid (`R=4` means 2x2 pooling per level), text and audio are 1D.
#
# **The pack.** A row is `[ text prefix | level 0 | level 1 | ... | level K-1 ]`. The prefix is the
# condition (prompt / caption / transcript, bytes, always fully known). Attention is *block-causal
# over levels*: a slot sees the prefix and every slot of levels <= its own, bidirectionally inside a
# level. The input at level *k* is the codes of level *k-1* upsampled to length *l_k*, so a level
# never sees its own targets (the leak that made an earlier version of this idea score below the
# constant baseline; the correctness cell asserts it is gone).
#
# **Codes** are 1..256 and 0 is NULL, so vocab = 257 and a real code can never be confused with padding.

# %%
from typing import Union


@dataclass
class Spec:
    K: int
    text: int            # finest text length (bytes)
    image: int           # finest image length (patches; a perfect square)
    audio: int           # finest audio length (mel frames)
    cond: int            # prefix length (bytes of prompt / caption / transcript)
    patch: int           # pixels per image patch side
    R: Union[int, dict] = 4   # one ratio for every modality, or {m: ratio}. A shared ratio is
                              # what gave text ONE level-0 code for 256 bytes (K=5, R=4).
    V: int = 257
    hop: int = 256
    sr: int = 16000
    n_mels: int = 80
    image_extra: int = 0 # extra residual levels at the finest image resolution (a 2nd codebook per position: +8 bits per slot of the finest grid)
    audio_extra: int = 0 # the same for audio: residual levels at the finest frame rate (8 codebooks in total = 7 extra)

    def __post_init__(self):
        # Resolved once; `self.R` keeps whatever was given so a scalar spec serialises as before.
        mods = ("text", "image", "audio")
        if isinstance(self.R, dict):
            missing = [m for m in mods if m not in self.R]
            if missing:
                raise KeyError(f"R is a per-modality dict but has no ratio for {missing}")
            self.Rm = {m: int(self.R[m]) for m in mods}
        else:
            self.Rm = {m: int(self.R) for m in mods}

    def R_of(self, m):
        """This modality's compression ratio. Every pooling site uses THIS, never `spec.R`."""
        return self.Rm[m]

    def lens(self, m):
        base = getattr(self, m)
        out = [base // self.R_of(m) ** (self.K - 1 - k) for k in range(self.K)]
        return out + [base] * {"image": self.image_extra, "audio": self.audio_extra}.get(m, 0)

    def ladder_report(self, min_level0=8):
        """-> (rows, warnings, fatal). A level-0 code that stands for a whole sample carries
        almost nothing and nothing downstream can learn it, so this is printed (and can block a
        run) BEFORE the data is encoded rather than diagnosed from a finished run.

        Measured, K=5/R=4 text (1/4/16/64/256 slots): level 0 was ONE code for 256 bytes and the
        probe gain at the next level was -0.31 bits, i.e. worse than a shuffled control. Images
        tolerate one slot (31.2% L0 accuracy measured - they are spatially redundant); a
        sequential modality does not.
        """
        rows, warn, fatal = [], [], []
        for m in ("text", "image", "audio"):
            base, R = getattr(self, m), self.R_of(m)
            lens = self.lens(m)[:self.K]
            span = base // max(1, lens[0])
            rows.append(f"{m:6s} R={R} base={base:5d} levels={lens} (L0: 1 code per {span})")
            if lens[0] < min_level0:
                how = "NCM_TEXT_R=2" if m == "text" else f"spec R[{m!r}]"
                msg = (f"{m}: level 0 has {lens[0]} slot(s) for {base} units (1 code per {span}) "
                       f"- below {min_level0}. Lower the ratio for {m} ({how}) or raise its base "
                       f"length to {min_level0 * R ** (self.K - 1)}+.")
                warn.append(msg)
                if m == "text":
                    fatal.append(msg)
        return rows, warn, fatal

    @property
    def side(self):                       # image side in pixels
        return math.isqrt(self.image) * self.patch

    @property
    def wave_len(self):                   # samples covered by `audio` frames
        return self.audio * self.hop

    def check(self):
        for m in ("text", "image", "audio"):
            assert getattr(self, m) % self.R_of(m) ** (self.K - 1) == 0, \
                f"{m} base {getattr(self, m)} not divisible by R^(K-1) = {self.R_of(m) ** (self.K - 1)}"
        assert math.isqrt(self.image) ** 2 == self.image
        assert math.isqrt(self.R_of("image")) ** 2 == self.R_of("image"), \
            "the image ratio must be a perfect square for 2D pooling"
        assert all(math.isqrt(n) ** 2 == n for n in self.lens("image"))
        return self


# Image bit budget (finest level, 8 bits per code): "v1" = 16x16 grid at 256 px = 2,048 bits/image (the original setting);
# "v2" = 32x32 grid at 256 px with two residual codebooks = 2 x 1,024 x 8 = 16,384 bits/image. v2 needs the learned image VQ
# (the DINO feature codec has a fixed 16x16 grid) and makes an image row ~7x longer (2,388 live codes against 341).
IMAGE_RES = _knob("IMAGE_RES", "v1")
_IMG = {"v1": dict(image=256, patch=16, image_extra=0), "v2": dict(image=1024, patch=8, image_extra=1)}[IMAGE_RES]
# Audio bit budget: "v1" = 512 frames x 8 bits = 4,096 bits/clip (~500 bit/s, about one DAC codebook); "v2" = 8 residual levels at the finest
# frame rate = 32,768 bits/clip (~4 kbps = 8 DAC codebooks, the setting that passed the listening test). v2 needs TOKENIZER="feat" (a residual
# quantiser on the DAC latents) and makes an audio row ~6x longer (about 4,266 live codes against 680).
AUDIO_RES = _knob("AUDIO_RES", "v1")
_AUDX = {"v1": 0, "v2": 7}[AUDIO_RES]

# Text's ladder ratio. 4 puts the whole 256-byte block in ONE level-0 code (nothing learnable
# there); 2 gives 16/32/64/128/256 and a level 0 that carries real structure. Costs a ~30%
# longer text row (624 packed slots against 469), which the token budget absorbs.
TEXT_R = _knob("TEXT_R", 2)
# Text context length in bytes (the FINEST level; the sample is that many bytes). Raising it
# raises the whole ladder proportionally at fixed R, so L0 keeps its 16-byte reach - but it
# also grows the finest level, and that level is predicted in ONE parallel forward with no
# byte-to-byte dependence, so more bytes per sample means more independently-wrong positions.
# 1024 is the sensible ceiling until the finest level gets a causal tail; 2048 mostly buys
# attention cost (T 4096) and cost the mix whole sources before TEXT_PACK.
TEXT_LEN = _knob("TEXT_LEN", 256)
# Audio window in mel frames (must be divisible by R**(K-1) = 256 at R=4, K=5; 1024 frames = 16.4 s
# at hop 256 / 16 kHz). This is not cosmetic: the largest open transcript corpus (People's Speech)
# segments at ~14.8 s (measured: p10 14.3, p90 15.0), and a window shorter than the segment drops
# the ENTIRE clip - at 512 frames only 1% of that corpus survived, which is why audio overfit while
# carrying weight 0.85. 1024 frames covers it; 2048 buys nothing but attention.
AUDIO_LEN = _knob("AUDIO_LEN", 1024)

PRESETS = {
    # same code path as "full", ~150M total / ~35M active: any 8 GB GPU, and CPU for CI
    "tiny": dict(d=512, layers=8, heads=8, kv_heads=2, ffn=1904, act_ckpt=False,
                 spec=dict(K=4, text=128, image=64, audio=512, cond=128, patch=8),
                 micro_tokens=8192, batch_codes=32768, workers=4, lr=1.5e-3, warmup=50,
                 fit_images=256, fit_audio=128, val_rows=16),
    # d=1536, L=16, expert h=3584 -> 264M per expert x 6 = 1.58B, trunk ~100M: ~1.68B total, ~360M active
    "full": dict(d=1536, layers=16, heads=16, kv_heads=4, ffn=3584, act_ckpt=True,
                 spec=dict(K=5, text=TEXT_LEN, audio=AUDIO_LEN, cond=128, audio_extra=_AUDX, **_IMG,
                           R=dict(text=TEXT_R, image=4, audio=4)),
                 micro_tokens=32768, batch_codes=1_000_000, workers=min(12, N_CPUS), lr=6e-4, warmup=1000,
                 fit_images=4096, fit_audio=1024, val_rows=64),
}
P = dict(PRESETS[PRESET])
SPEC = Spec(**P["spec"]).check()
# The ladder gate. Printed always; a degenerate TEXT level 0 blocks the run, because it is the
# one defect that no amount of training, capacity or data can reach past.
LADDER_GATE = _knob("LADDER_GATE", PRESET == "full")
_lrows, _lwarn, _lfatal = SPEC.ladder_report()
print("ladder:")
for _r in _lrows:
    print("  " + _r)
if _lwarn:
    print("  !! coarse level 0 (1 code per sample-scale span):")
    for _w in _lwarn:
        print("     " + _w)
if _lfatal and LADDER_GATE:
    raise SystemExit("DEGENERATE TEXT LADDER - refusing to encode a corpus nothing can learn from.\n"
                     "  " + _lfatal[0] + "\n"
                     "  Fix: NCM_TEXT_R=2 (default for preset 'full'), or NCM_LADDER_GATE=0 to override.")
print(f"  audio: {SPEC.audio} frames = {SPEC.wave_len / SPEC.sr:.1f} s per clip, window = spec.wave_len "
      f"(a clip longer than this is DROPPED, not cropped)")
_tl = SPEC.lens("text")
print(f"  text: {SPEC.text} bytes/sample; the finest level is {_tl[-1]} bytes in ONE parallel "
      f"forward (its parent supplies 1 code per {_tl[-1] // max(1, _tl[-2])} bytes, neighbours nothing)")
if SPEC.text >= 1024 and not _knob("TEXT_PACK", 1):
    print("  !! TEXT_PACK=0 with a long text sample: every document shorter than "
          f"{SPEC.text // 2} bytes is dropped and a document yields at most 4 samples")
if SPEC.text >= 1024:
    print(f"  note: the longest text row is {SPEC.text + SPEC.cond} packed slots; if the first "
          f"step OOMs, halve NCM_MICRO_TOKENS (32768 -> 16384) and expect attention to take a "
          f"larger share of the step than the 9% measured at text=256")
N_GROUP_EXPERTS = 2
_BIG_GPU = PRESET == "full" and VRAM_GB >= 90        # RTX PRO 6000 (95 GB), MI300X (192 GB): 69 GiB peak measured with micro 49152 and no activation checkpointing
# A long text or audio sample is the same memory problem as a v2 residual image row: more packed
# slots per row, so the v1 micro-batch measurements (50-69 GiB) no longer bound the peak. Grouping
# them here keeps the big-GPU shortcuts (micro 49152, no activation checkpointing) OFF, which is
# what stops a step-1 OOM on a 40-95 GiB card.
_LONG_ROWS = IMAGE_RES != "v1" or AUDIO_RES != "v1" or TEXT_LEN > 256 or AUDIO_LEN > 512   # v2 image / audio rows are 4-6x longer with up to 12 levels: the v1 memory measurements below do not apply (an audio micro-batch ran out of memory on 95 GB)
if PRESET == "full" and VRAM_GB >= 70 and ACT_CKPT < 0 and not _LONG_ROWS:
    ACT_CKPT = 0                                      # no activation checkpointing: faster. 50 GiB peak measured at micro 32768, so it fits an 80 GB H100/A100 too
MICRO_TOKENS = MICRO_TOKENS or (49152 if _BIG_GPU and not _LONG_ROWS else P["micro_tokens"])   # +2% codes/s over 32768 on the RTX PRO 6000 (fuller GEMMs)
BATCH_CODES = BATCH_CODES or P["batch_codes"]
WORKERS = WORKERS or P["workers"]
FIT_IMAGES = FIT_IMAGES or P["fit_images"]
FIT_AUDIO = FIT_AUDIO or P["fit_audio"]
VAL_ROWS = VAL_ROWS or P["val_rows"]
LR = LR or P["lr"]
WARMUP = WARMUP or P["warmup"]

ROW_TYPES = ("text0", "text", "image", "audio")      # text0 = plain text, text = prompt -> response
ROW_MOD = {"text0": "text", "text": "text", "image": "image", "audio": "audio"}
ROW_COND = {"text0": 0, "text": SPEC.cond, "image": SPEC.cond, "audio": SPEC.cond}


class Layout:
    """Static geometry of one row type: where every slot lives and which group routes it."""

    def __init__(self, spec, rtype):
        self.rtype, self.mod, self.cond = rtype, ROW_MOD[rtype], ROW_COND[rtype]
        self.lens = spec.lens(self.mod)
        self.K = len(self.lens)                              # = spec.K, +image_extra for images
        self.lvl, o = [], self.cond
        for n in self.lens:
            self.lvl.append((o, o + n)); o += n
        self.T, self.Tt = o, o - self.cond                   # packed length, target length
        bid = [0] * self.cond + [k + 1 for k, n in enumerate(self.lens) for _ in range(n)]
        self.block_id = torch.tensor(bid, dtype=torch.int32)
        self.mask = self.block_id[None, :] <= self.block_id[:, None]    # [q, kv] True = may attend
        # which MoE group routes which slots (expert-choice runs per segment = per level)
        self.segs = {}
        if self.cond:
            self.segs.setdefault("text", []).append((0, self.cond))
        self.segs.setdefault(self.mod, []).extend(self.lvl)

    def to(self, device):
        self.block_id, self.mask = self.block_id.to(device), self.mask.to(device)
        return self


LAYOUTS = {r: Layout(SPEC, r) for r in ROW_TYPES}
for _r, _l in LAYOUTS.items():
    print(f"{_r:6s} T={_l.T:5d} target={_l.Tt:5d} live codes/row={_l.Tt} levels={_l.lens}")
_gi = math.isqrt(SPEC.image)
print(f"BIT BUDGET (finest resolution, 8 bits per code): "
      f"image {SPEC.image * (1 + SPEC.image_extra) * 8:,} bits/image = {_gi}x{_gi} grid x {1 + SPEC.image_extra} codebook(s) x 8 ({SPEC.side}px, IMAGE_RES={IMAGE_RES}) | "
      f"audio {SPEC.audio * (1 + SPEC.audio_extra) * 8:,} bits/clip = {SPEC.audio * (1 + SPEC.audio_extra) * 8 / (SPEC.wave_len / SPEC.sr):.0f} bit/s ({SPEC.wave_len / SPEC.sr:.1f} s, {1 + SPEC.audio_extra} level(s) at the finest rate, AUDIO_RES={AUDIO_RES}) | text {SPEC.text * 8:,} bits/row (lossless bytes)")
