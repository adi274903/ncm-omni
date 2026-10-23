"""Codecs: raw text/image/audio/video -> K nested code levels. Frozen, never trained here.

Hard contract, the one thing this file exists to guarantee:

    encode(obj) -> list of int arrays, one per level, with len(level_k) == schedule.level_len(m, k)

Every level is produced by *pooling the finest representation*, never by independently
re-encoding at a different size. That is what makes level k "compression R^(K-1-k)" instead of
"level k is a different thumbnail", and it is why the cross-modal level alignment means
something. Violations are caught in producer.encode_sample, which logs and masks the modality.

Tiers:
  TextByteCodec        UTF-8 bytes at the finest level (exact), hashed spans coarser
  ImagePyramidCodec    patch-mean pyramid, 6 bins/channel -> <256 codes, decodable (PSNR real)
  VideoPyramidCodec    frame x patch pyramid, same pooling ladder, decodable per frame
  AudioMelPyramidCodec log-mel frames quantised against a FIXED seeded codebook (1 code/frame)
  SpanHashCodec        no dependencies, no semantics: plumbing test only

The image/audio/video codebooks here are stand-ins, and the README says so. The ladder geometry
is the real thing; the codebooks are what a trained VQ (NCM text RVQ, Infinity image VQ, DAC
audio) replaces, and `build_codec` is the single place to swap them.
"""
import hashlib

import numpy as np


class Codec:
    """Interface. K / R / base_len must match the schedule the producer was configured with."""
    name = "codec"
    kind = "synthetic"
    exact_at_finest = False
    decodable = False

    def __init__(self, K=6, R=2, codebook_size=256, base_len=64):
        self.K, self.R, self.V = K, R, codebook_size
        self.base_len = base_len

    def level_len(self, k):
        return max(1, round(self.base_len / self.R ** (self.K - 1 - k)))

    def encode(self, obj):
        raise NotImplementedError

    def decode(self, codes):
        raise NotImplementedError

    def available(self):
        return True

    def info(self):
        return dict(name=self.name, kind=self.kind, K=self.K, R=self.R, V=self.V,
                    base_len=self.base_len, decodable=self.decodable,
                    exact_at_finest=self.exact_at_finest)


# ------------------------------------------------------------------ shared machinery
def pool_feats(feat: np.ndarray, l: int) -> np.ndarray:
    """[n, ...] -> [l, ...] by mean-pooling contiguous groups; resample if not divisible."""
    n = len(feat)
    if l == n:
        return feat
    if n == 0:
        return np.zeros((l,) + feat.shape[1:], dtype=np.float32)
    if l < n and n % l == 0:
        return feat.reshape(l, n // l, *feat.shape[1:]).mean(1)
    idx = np.linspace(0, n - 1, l).round().astype(int)          # upsampling or non-integer ratio
    return feat[idx]


def quant_rgb(feat: np.ndarray, V: int) -> np.ndarray:
    """[l, 3] in [0,1] -> [l] codes in 1..B^3. 1-based: index 0 is the NULL slot, never a code."""
    B = max(2, int(round(V ** (1 / 3))))
    q = np.clip((np.asarray(feat, dtype=np.float32) * B).astype(np.int32), 0, B - 1)
    return (q[..., 0] * B * B + q[..., 1] * B + q[..., 2] + 1).astype(np.int32)


def dequant_rgb(codes, V: int) -> np.ndarray:
    B = max(2, int(round(V ** (1 / 3))))
    c = np.asarray(codes, dtype=np.int32) - 1
    return np.stack([c // (B * B), (c // B) % B, c % B], -1).astype(np.float32) / B


def patch_means(img, grid: int) -> np.ndarray:
    """PIL image -> [grid*grid, 3] patch means in [0,1]."""
    g = np.asarray(img.convert("RGB").resize((grid * 8, grid * 8), resample=2),
                   dtype=np.float32) / 255.0
    return g.reshape(grid, 8, grid, 8, 3).mean(axis=(1, 3)).reshape(-1, 3)


# ------------------------------------------------------------------ text
class SpanHashCodec(Codec):
    """Any bytes -> deterministic codes per level. No semantics: validates plumbing only."""
    kind, name = "synthetic", "spanhash"

    def __init__(self, K=6, R=2, codebook_size=256, base_len=64, seed=0):
        super().__init__(K, R, codebook_size, base_len)
        self.seed = seed

    def _codes(self, buf: bytes, l: int, k: int) -> np.ndarray:
        if len(buf) == 0:
            return np.ones(l, dtype=np.int32)                 # 1 = the first real code, not NULL
        span = max(1, int(np.ceil(len(buf) / l)))
        out = np.empty(l, dtype=np.int32)
        for i in range(l):
            chunk = buf[i * span:(i + 1) * span]
            h = hashlib.blake2b(bytes([k]) + chunk, digest_size=8,
                                salt=self.seed.to_bytes(8, "little")).digest()
            out[i] = int.from_bytes(h, "little") % self.V + 1
        return out

    def encode(self, obj):
        raw = obj if isinstance(obj, (bytes, bytearray)) else str(obj).encode("utf-8", "ignore")
        return [self._codes(raw, self.level_len(k), k) for k in range(self.K)]


class TextByteCodec(Codec):
    """UTF-8 bytes at the finest level; coarser levels hash R^k spans.

    The finest level decodes exactly, which is the number worth reporting (the paper's RVQ
    tokenizer gets ~99.99% exact text reconstruction; bytes get 100% by construction, so the
    honest claim here is only "the bottom of the ladder is lossless").
    """
    kind, name, exact_at_finest, decodable = "real", "text_byte", True, True

    def __init__(self, K=6, R=2, codebook_size=256, base_len=64, seed=1):
        super().__init__(K, R, codebook_size, base_len)
        self.seed = seed

    def encode(self, text):
        raw = (text or "").encode("utf-8", "ignore")[: self.base_len]
        hasher = SpanHashCodec(self.K, self.R, self.V, self.base_len, self.seed)
        out = []
        for k in range(self.K):
            l = self.level_len(k)
            if k == self.K - 1:                        # lossless bottom, shifted to 1..256
                b = np.zeros(l, dtype=np.int32)
                b[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)[:l].astype(np.int32) + 1
                out.append(b)
            else:
                out.append(hasher._codes(raw, l, k))
        return out

    def decode(self, codes):
        b = np.clip(np.asarray(codes[-1], dtype=np.int32) - 1, 0, 255).astype(np.uint8)
        return b.tobytes().rstrip(b"\x00").decode("utf-8", "ignore")


# ------------------------------------------------------------------ image
class ImagePyramidCodec(Codec):
    """PIL image -> patch-mean pyramid. Level k pools the finest grid by R^(K-1-k).

    decode() inverts the finest level, so per-level PSNR against the real image is a number you
    can report rather than a placeholder.
    """
    kind, name, decodable = "real", "image_pyramid", True

    def __init__(self, K=6, R=2, codebook_size=256, base_len=64):
        super().__init__(K, R, codebook_size, base_len)
        self.grid = max(1, int(round(np.sqrt(base_len))))

    def _feats(self, img):
        return patch_means(img, self.grid)                       # [grid*grid, 3]

    def encode(self, img):
        f = self._feats(img)
        return [quant_rgb(pool_feats(f, self.level_len(k)), self.V) for k in range(self.K)]

    def decode(self, codes, side=None):
        px = dequant_rgb(codes[-1], self.V)                      # [grid*grid, 3]
        g = side or self.grid
        return px[: g * g].reshape(g, g, 3)


# ------------------------------------------------------------------ video
class VideoPyramidCodec(Codec):
    """Frames -> (frame x patch) pyramid. Level k pools *both* the frame rate and the patch grid.

    Video is data-path only in this project, but it has to obey the same ladder: level k of a
    clip must be the clip at R^(K-1-k) the spatio-temporal detail, or its slots are not
    comparable to the image slots at the same level.
    """
    kind, name, decodable = "real", "video_pyramid", True

    def __init__(self, K=6, R=2, codebook_size=256, base_len=32, n_frames=8, grid=2):
        super().__init__(K, R, codebook_size, base_len)
        self.n_frames, self.grid = n_frames, grid
        assert n_frames * grid * grid == base_len, "frames x patches must equal base_len"

    def _feats(self, frames):
        pick = np.linspace(0, len(frames) - 1, self.n_frames).round().astype(int)
        feats = [patch_means(frames[i], self.grid) for i in pick]
        return np.concatenate(feats, 0)                          # [n_frames*grid^2, 3]

    def encode(self, frames):
        f = self._feats(frames)
        return [quant_rgb(pool_feats(f, self.level_len(k)), self.V) for k in range(self.K)]

    def decode(self, codes):
        px = dequant_rgb(codes[-1], self.V)                      # [n_frames*grid^2, 3]
        return px.reshape(self.n_frames, self.grid, self.grid, 3)


# ------------------------------------------------------------------ audio
def _mel_fb(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """[n_fft//2+1, n_mels] triangular mel filterbank (HTK scale)."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    hz = mel2hz(np.linspace(hz2mel(0.0), hz2mel(sr / 2), n_mels + 2))
    bins = np.clip(np.floor((n_fft + 1) * hz / sr).astype(int), 0, n_fft // 2)
    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], max(bins[i] + 1, bins[i + 1]), max(bins[i + 1] + 2, bins[i + 2])
        l, c, r = min(l, n_fft // 2), min(c, n_fft // 2), min(r, n_fft // 2 + 1)
        if c > l:
            fb[l:c, i] = np.linspace(0, 1, c - l, endpoint=False)
        if r > c:
            fb[c:r, i] = np.linspace(1, 0, r - c, endpoint=False)
    return fb


def _mel_numpy(x: np.ndarray, sr, n_mels=32, n_fft=400, hop=160) -> np.ndarray:
    """Log-mel without torchaudio, so a torch/torchaudio version skew cannot kill the pipeline."""
    if sr and sr != 16000:
        n = max(1, int(round(len(x) * 16000 / sr)))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) ** 2
    return np.log(spec @ _mel_fb(n_mels, n_fft, 16000) + 1e-6).T          # [n_mels, T]


class AudioMelPyramidCodec(Codec):
    """Waveform -> log-mel frames, one code per frame, levels = frame-rate pyramid.

    Audio's hierarchy is temporal *rate*, not spatial resolution: level k is the spectrogram at
    1/R^(K-1-k) the frame rate. One code per frame means the codebook has to carry a whole mel
    column, so this uses a fixed seeded random codebook (nearest prototype in normalised mel
    space). The ladder is real; the codebook is a stand-in for DAC/EnCodec.
    """
    kind, name, decodable = "real", "audio_mel_pyramid", True

    def __init__(self, K=6, R=2, codebook_size=256, base_len=96, n_mels=32, seed=7):
        super().__init__(K, R, codebook_size, base_len)
        self.n_mels = n_mels
        g = np.random.default_rng(seed)
        C = g.standard_normal((n_mels, codebook_size)).astype(np.float32)
        self.C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-8)     # [n_mels, V]

    def _mel(self, obj):
        wav, sr = obj
        x = np.asarray(wav, dtype=np.float32).flatten()
        if x.size == 0:
            return np.zeros((self.n_mels, 1), dtype=np.float32)
        try:
            import torch
            import torchaudio
            w = torch.as_tensor(x)
            if sr and sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            mel = torchaudio.transforms.MelSpectrogram(
                16000, n_fft=400, hop_length=160, n_mels=self.n_mels)(w)
            return torch.log(mel + 1e-6).numpy()                            # [n_mels, T]
        except Exception:                          # torchaudio missing or built for another torch
            return _mel_numpy(x, sr, n_mels=self.n_mels)

    def encode(self, obj):
        mel = self._mel(obj).T                                             # [T, n_mels]
        out = []
        for k in range(self.K):
            x = pool_feats(mel, self.level_len(k))                         # [l, n_mels]
            x = x - x.mean(0, keepdims=True)
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
            out.append((x @ self.C).argmax(1).astype(np.int32) + 1)        # 1..V, never NULL
        return out

    def decode(self, codes):
        idx = np.clip(np.asarray(codes[-1], dtype=np.int32) - 1, 0, self.V - 1)
        return self.C[:, idx].T                                            # [l, n_mels]


def build_codec(modality, K, R, V, base_len, prefer_real=True):
    """Pick the strongest codec that is importable here. Never silently pretend."""
    if modality == "text":
        return TextByteCodec(K, R, V, base_len) if prefer_real else SpanHashCodec(K, R, V, base_len)
    if modality == "image":
        return ImagePyramidCodec(K, R, V, base_len)
    if modality == "video":
        grid = 2 if base_len % 4 == 0 else 1
        frames = max(2, base_len // (grid * grid))
        return VideoPyramidCodec(K, R, V, frames * grid * grid, n_frames=frames, grid=grid)
    if modality == "audio":
        return AudioMelPyramidCodec(K, R, V, base_len)
    return SpanHashCodec(K, R, V, base_len)
