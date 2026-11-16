# %% [markdown]
# ## 3. Frozen codecs (all on the GPU)
#
# No learned encoder: every modality is turned into the **same ladder** of 256-way codes by a fixed
# procedure, so "level k" is comparable across text, image and audio.
#
# | modality | finest level (K-1) | coarser levels | decode |
# |---|---|---|---|
# | text | the byte itself (exact) | span-pooled random byte projection, sign bits = 8-bit locality-sensitive code | bytes -> UTF-8 |
# | image | VQ of a `patch x patch x 3` pixel block | 2x2 average-pooled grid (a blurry image), VQ per level | codeword tiles |
# | audio | VQ of an 80-bin log-mel frame (16 ms hop) | rate pyramid (average-pool in time), VQ per level | log-mel -> pseudo-inverse -> Griffin-Lim |
#
# Codebooks are **fit once at startup** by k-means on a streamed sample that is held in RAM and then
# discarded; they are tiny and live in the checkpoint, so a resumed run uses the same codes.
# Quality of image/audio generations is bounded by these codecs: they are coarse by construction.

# %%
class TextCodec:
    def __init__(self, spec, device, seed=1234):
        self.spec, self.device = spec, device
        g = torch.Generator().manual_seed(seed)
        self.planes = torch.randn(256, 8, generator=g).to(device)              # byte -> 8 hash planes
        self.pw = (1 << torch.arange(8, device=device)).long()

    @torch.no_grad()
    def encode(self, b):
        """b [B, L] uint8 -> K long tensors [B, l_k] with codes in 1..256."""
        K, R = self.spec.K, self.spec.R_of("text")
        b = b.long()
        codes = [None] * K
        codes[K - 1] = b + 1
        proj = self.planes[b].transpose(1, 2)                                   # [B, 8, L]
        for j in range(K - 1):
            pooled = F.avg_pool1d(proj, R ** (K - 1 - j)).transpose(1, 2)       # [B, l_j, 8]
            codes[j] = 1 + ((pooled > 0).long() * self.pw).sum(-1)
        return codes

    @staticmethod
    def decode(finest):
        return [bytes((c - 1).clamp(0, 255).to(torch.uint8).cpu().tolist()).rstrip(b"\x00").decode("utf-8", "replace")
                for c in finest]


def kmeans(x, k=256, iters=25, seed=0):
    """x [N, D] float -> centroids [k, D]. Lloyd on the GPU with dead-centroid re-seeding."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    if x.shape[0] > 200_000:
        x = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:200_000]]
    N = x.shape[0]
    if N < k:
        x = torch.cat([x, x[torch.randint(N, (k - N,), generator=g, device=x.device)]
                       + 1e-3 * torch.randn(k - N, x.shape[1], device=x.device, generator=g)])
        N = k
    cent = x[torch.randperm(N, generator=g, device=x.device)[:k]].clone()
    x2 = (x * x).sum(1, keepdim=True)
    for _ in range(iters):
        a = (x2 - 2 * x @ cent.T + (cent * cent).sum(1)).argmin(1)
        sums = torch.zeros_like(cent).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        new = sums / cnt.clamp(min=1)[:, None]
        dead = cnt == 0
        if dead.any():
            new[dead] = x[torch.randint(N, (int(dead.sum()),), generator=g, device=x.device)]
        cent = new
    return cent


def sample_points(feats, cap):
    """Keep at most `cap` random rows of each level's features, so fitting never holds a whole corpus."""
    out = []
    for f in feats:
        f = f.reshape(-1, f.shape[-1])
        out.append(f if f.shape[0] <= cap else f[torch.randperm(f.shape[0], device=f.device)[:cap]])
    return out


def vq(x, cb):
    """x [..., D], cb [k, D] -> 1-based codes [...]. argmin ||x-c||^2 via one matmul."""
    sh = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    d = (cb * cb).sum(1)[None, :] - 2 * xf @ cb.T
    return (d.argmin(1) + 1).view(sh)


class ImageCodec:
    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.G, self.p = math.isqrt(spec.image), spec.patch
        self.D = 3 * self.p * self.p
        self.cb = None                                                           # list K x [256, D]

    def features(self, img):
        """img uint8 [B, 3, S, S] -> list K of float [B, n_k, D]; level j is the grid pooled R^(K-1-j)x."""
        K, f = self.spec.K, math.isqrt(self.spec.R_of("image"))
        x = img.float() / 255.0 - 0.5
        B = x.shape[0]
        cur = F.unfold(x, self.p, stride=self.p).view(B, self.D, self.G, self.G)
        feats = [None] * K
        for j in reversed(range(K)):
            feats[j] = cur.flatten(2).transpose(1, 2).contiguous()
            if j > 0:
                cur = F.avg_pool2d(cur, f)
        return feats

    @torch.no_grad()
    def fit(self, feats_list):
        """feats_list: per batch, a list over levels of [m, D] points (see `sample_points`); one codebook per level."""
        self.cb = [kmeans(torch.cat([f[j].reshape(-1, self.D) for f in feats_list]), 256, seed=j)
                   for j in range(self.spec.K)]

    @torch.no_grad()
    def encode(self, img):
        return [vq(f, self.cb[j]) for j, f in enumerate(self.features(img))]

    @torch.no_grad()
    def decode(self, codes, level=None):
        """codes [B, n_j] at `level` (default finest), or the whole code list (then `codes[level]` is used) -> uint8 [B, 3, S, S]."""
        level = self.spec.K - 1 if level is None else level
        if isinstance(codes, (list, tuple)):
            codes = codes[level]
        cells = self.cb[level][(codes - 1).clamp(min=0)]                         # [B, n, D]
        g = math.isqrt(codes.shape[1])
        img = F.fold(cells.transpose(1, 2), (g * self.p, g * self.p), self.p, stride=self.p)
        S = self.spec.side
        if img.shape[-1] != S:
            img = F.interpolate(img, size=(S, S), mode="nearest")
        return ((img + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)

    def state(self):
        return None if self.cb is None else [c.cpu() for c in self.cb]

    def load(self, st):
        self.cb = [c.to(self.device) for c in st]


def decode_jpeg_gpu(blobs, side, device):
    """list of JPEG byte strings -> uint8 [B, 3, side, side] (short side resized, centre-cropped), nvJPEG if available."""
    from torchvision.io import decode_jpeg, ImageReadMode
    ts = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs]
    try:
        imgs = decode_jpeg(ts, device=device, mode=ImageReadMode.RGB)
    except Exception:
        imgs = [decode_jpeg(t, mode=ImageReadMode.RGB).to(device) for t in ts]
    out = []
    for im in imgs:
        _, H, W = im.shape
        s = side / min(H, W)
        nh, nw = max(side, round(H * s)), max(side, round(W * s))
        im = F.interpolate(im[None].float(), size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)[0]
        t, l = (nh - side) // 2, (nw - side) // 2
        out.append(im[:, t:t + side, l:l + side].round().clamp(0, 255).to(torch.uint8))
    return torch.stack(out)


def _mel_fb(n_freqs, n_mels, sr):
    """Slaney-style mel filterbank [n_freqs, n_mels] (no torchaudio dependency)."""
    def hz2mel(f):
        f = torch.as_tensor(f, dtype=torch.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + torch.log(f.clamp(min=1000.0) / 1000.0) / (math.log(6.4) / 27.0)
        return torch.where(f >= 1000.0, log, lin)
    def mel2hz(m):
        m = torch.as_tensor(m, dtype=torch.float64)
        lin = (200.0 / 3) * m
        log = 1000.0 * torch.exp((math.log(6.4) / 27.0) * (m - 15.0))
        return torch.where(m >= 15.0, log, lin)
    all_f = torch.linspace(0, sr / 2, n_freqs, dtype=torch.float64)
    pts = mel2hz(torch.linspace(float(hz2mel(0.0)), float(hz2mel(sr / 2)), n_mels + 2, dtype=torch.float64))
    diff = pts[1:] - pts[:-1]
    slopes = pts[None, :] - all_f[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    fb = fb * (2.0 / (pts[2:n_mels + 2] - pts[:n_mels]))[None, :]
    return fb.float()


class AudioCodec:
    N_FFT = 1024

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.fb = _mel_fb(self.N_FFT // 2 + 1, spec.n_mels, spec.sr).to(device)    # [513, 80]
        self.pinv = torch.linalg.pinv(self.fb.cpu()).to(device)                    # [80, 513]
        self.win = torch.hann_window(self.N_FFT, device=device)
        self.mean, self.std = -5.0, 2.5
        self.cb = None

    def logmel(self, wave):
        """wave float [B, n] -> log-mel [B, audio, 80] (n = audio * hop)."""
        s = torch.stft(wave, self.N_FFT, self.spec.hop, window=self.win, center=True, return_complex=True)
        mel = (s.abs().pow(2).transpose(1, 2) @ self.fb)[:, :self.spec.audio]
        return torch.log(mel.clamp(min=1e-5))

    def features(self, wave):
        K, R = self.spec.K, self.spec.R_of("audio")
        x = ((self.logmel(wave) - self.mean) / self.std).transpose(1, 2)          # [B, 80, F]
        return [F.avg_pool1d(x, R ** (K - 1 - j)).transpose(1, 2).contiguous() if j < K - 1
                else x.transpose(1, 2).contiguous() for j in range(K)]

    @torch.no_grad()
    def set_stats(self, waves):
        lm = torch.cat([self.logmel(w) for w in waves]).flatten()
        self.mean, self.std = float(lm.mean()), float(lm.std().clamp(min=1e-3))

    @torch.no_grad()
    def fit(self, feats_list):
        self.cb = [kmeans(torch.cat([f[j].reshape(-1, self.spec.n_mels) for f in feats_list]), 256, seed=j)
                   for j in range(self.spec.K)]

    @torch.no_grad()
    def encode(self, wave):
        return [vq(f, self.cb[j]) for j, f in enumerate(self.features(wave))]

    @torch.no_grad()
    def decode_mel(self, codes, level=None):
        """codes [B, n_level] (or the whole code list) -> natural-log mel [B, audio, 80] (codeword lookup, repeated up to the finest rate)."""
        level = self.spec.K - 1 if level is None else level
        if isinstance(codes, (list, tuple)):
            codes = codes[level]
        z = self.cb[level][(codes - 1).clamp(min=0)]                              # [B, n, 80]
        if z.shape[1] != self.spec.audio:
            z = z.repeat_interleave(self.spec.audio // z.shape[1], 1)
        return z * self.std + self.mean

    @torch.no_grad()
    def mel_to_mag(self, mel, iters=None):
        """Power mel [B, F, 80] -> magnitude spectrum [B, 513, F].

        `pinv` (the old path) goes negative, the clamp then leaves spectral holes and Griffin-Lim sounds metallic. `nnls` solves
        S >= 0 with S @ fb ~ mel by multiplicative KL updates (the NMF rule), so the spectrum stays positive and fills the holes."""
        if AUDIO_INV == "pinv":
            return (mel @ self.pinv).clamp(min=1e-8).sqrt().transpose(1, 2)
        W = self.fb                                                               # [513, 80]
        S = ((mel / W.sum(0).clamp(min=1e-8)) @ W.T).clamp(min=1e-8)              # positive back-projection as the start
        den = W.sum(1).clamp(min=1e-8)                                            # [513]
        for _ in range(iters or AUDIO_NNLS_ITERS):
            S = S * (((mel / (S @ W).clamp(min=1e-8)) @ W.T) / den)
        return S.clamp(min=1e-8).sqrt().transpose(1, 2)

    @torch.no_grad()
    def decode(self, codes, level=None, gl_iters=64):
        """codes [B, F] -> waveform float [B, n]: codeword log-mel -> power spec (NNLS) -> Griffin-Lim."""
        mag = self.mel_to_mag(torch.exp(self.decode_mel(codes, level)))
        return griffin_lim(mag, self.win, self.spec.hop, self.N_FFT, gl_iters)

    def state(self):
        return None if self.cb is None else dict(cb=[c.cpu() for c in self.cb], mean=self.mean, std=self.std)

    def load(self, st):
        self.cb, self.mean, self.std = [c.to(self.device) for c in st["cb"]], st["mean"], st["std"]


@torch.no_grad()
def griffin_lim(mag, win, hop, n_fft, iters=64, momentum=0.99):
    """Fast Griffin-Lim. mag [B, n_freq, F] -> waveform [B, (F-1)*hop]."""
    g = torch.Generator(device=mag.device).manual_seed(0)
    phase = torch.exp(2j * math.pi * torch.rand(mag.shape, device=mag.device, generator=g))
    spec, prev = mag * phase, None
    for _ in range(iters):
        wav = torch.istft(spec, n_fft, hop, window=win, center=True)
        est = torch.stft(wav, n_fft, hop, window=win, center=True, return_complex=True)[..., :mag.shape[-1]]
        if prev is not None:
            est = est + momentum * (est - prev)
        prev = est
        spec = mag * est / est.abs().clamp(min=1e-8)
    return torch.istft(spec, n_fft, hop, window=win, center=True)


class Codecs:
    def __init__(self, spec, device):
        if TOKENIZER == "feat":                     # frozen DINOv2 / DAC features + a data-fit text ladder (section 3b)
            self.text = make_text_codec(spec, device)
            self.image = VQImageCodec(spec, device) if IMAGE_CODEC == "vq" else DinoImageCodec(spec, device, FEAT_QUANT)    # IMAGE_CODEC="vq": learned pixel codec (needed for IMAGE_RES="v2")
            self.audio = (DacResidualAudioCodec if AUDIO_RES == "v2" else DacAudioFeatureCodec)(spec, device, FEAT_QUANT)
        elif TOKENIZER == "vq":                     # learned multi-scale residual VQ-VAEs (defined in the next section, trained at startup)
            self.text, self.image, self.audio = TextCodec(spec, device), VQImageCodec(spec, device), VQAudioCodec(spec, device)
        else:
            self.text, self.image, self.audio = TextCodec(spec, device), ImageCodec(spec, device), AudioCodec(spec, device)

    def ready(self):
        return self.image.cb is not None and self.audio.cb is not None

    def state(self):
        d = dict(image=self.image.state(), audio=self.audio.state())
        if hasattr(self.text, "state") and self.text.state() is not None:       # only the data-fit text codecs have a state
            d["text"] = self.text.state()
        return d

    def load(self, st):
        self.image.load(st["image"]); self.audio.load(st["audio"])
        if st.get("text") is not None and hasattr(self.text, "load"):
            self.text.load(st["text"])
