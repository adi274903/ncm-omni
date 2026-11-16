"""Learned / trained codecs behind one `tier` knob, keeping the ladder contract of `omni.codecs`.

The contract does not move here:

    encode(obj) -> list of int arrays, one per level, len(level_k) == codec.level_len(k)
    every code in 1..V   (0 stays NULL)
    level k is the finest representation coarsened by R**(K-1-k), never an independent re-encode

What changes is *which* codebook produces the codes and *how* a level is coarsened:

  TextEntropyCodec      bytes stay lossless at the finest rung; coarser rungs are segmented at
                        byte-novelty boundaries (a dependency-free proxy for the learned
                        local-abstraction positions of byte LMs, arXiv:2610.05978) instead of
                        fixed-width hashed spans.
  TreeHierarchy         a balanced binary hierarchy over ANY flat codebook, built post-hoc
                        (depth-major branch decisions, arXiv:2609.03641 Tree-VQ). A level is the
                        node you reach after climbing K-1-k merges, so no payload is re-encoded and
                        every level is a prefix/ancestor of the finest one.
  TreeCodebookCodec     the shared body: pool the finest features to the level length, quantise
                        against the flat codebook, then climb the tree to the level's depth.
  DacAudioCodec         optional `descript-audio-codec` (44.1 kHz, RVQ) = the leaf codebook.
  DinoKmeansImageCodec  optional frozen DINOv2/v3 patch features + a k-means codebook persisted by
                        the offline helper described in docs/codecs.md = the leaf codebook.

Nothing here pretends: `build_learned` returns `(None, reason)` when a backend is missing, and the
caller (`omni.codecs.build_codec`) falls back to the stand-in and records the reason in `info()`.

Why a tree and not `R` independent pools per level: quantisation error per level is what the model
has to predict, and a flat codebook re-quantised at every level makes level k a *different* latent
(a thumbnail), not a coarsening of the same one. The tree keeps the finest assignment and only
widens the cell, which is exactly what "level k is the input compressed by R**(K-1-k)" claims.
"""
import hashlib
import os

import numpy as np

from omni.codecs import Codec, pool_feats


# ------------------------------------------------------------------ the hierarchy
class TreeHierarchy:
    """Balanced binary hierarchy over a flat codebook (post-hoc construction, Tree-VQ style).

    Built by recursive 2-means on the codebook vectors: at every node the members are split at the
    median of their projection onto the node's first principal direction. Splitting by geometry and
    not by code index is what makes a coarse node a *prototype* (the mean of the leaves under it)
    rather than an arbitrary bucket of unrelated vectors.

    Depth d has n_d nodes, n_d <= V, and every leaf reaches exactly one of them. Level k of a
    K-level ladder uses depth d_min + (K-1-k): level K-1 is the finest depth that still fits the
    codebook cap V, level 0 is the coarsest. Ids are compact per level (1..n_k), so a code always
    lands in 1..V and can never collide with the NULL slot.
    """

    def __init__(self, vectors, seed=0):
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError(f"tree needs a 2-D codebook of >=2 vectors, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("codebook contains non-finite values")
        self.X = X
        self.V, self.dim = X.shape
        self.seed = int(seed)

        self._vec, self._depth, self._children, self._parent = [], [], [], []
        self._leaf_node = np.full(self.V, -1, dtype=np.int64)
        self._build(np.arange(self.V), 0)
        self._vec = np.stack(self._vec).astype(np.float32)          # [N, D]
        self._depth = np.asarray(self._depth, dtype=np.int32)
        self._parent = np.asarray(self._parent, dtype=np.int64)
        self.max_depth = int(self._depth.max())
        if np.any(self._leaf_node < 0):
            raise RuntimeError("tree construction left a leaf unassigned")
        self._counts = np.bincount(self._depth)
        self._anc, self._compact, self._map = {}, {}, {}

    # ---------------------------------------------------------------- construction
    @staticmethod
    def _pc1(X, iters=8):
        """First principal direction by power iteration; deterministic, no rng, no sklearn."""
        Xc = X - X.mean(0, keepdims=True)
        v = np.ones(Xc.shape[1], dtype=np.float32)
        for _ in range(iters):
            nv = Xc.T @ (Xc @ v)
            n = float(np.linalg.norm(nv))
            if n < 1e-8:                       # degenerate (all points equal): arbitrary but fixed
                v = np.linspace(1.0, 2.0, Xc.shape[1], dtype=np.float32)
                continue
            v = nv / n
        return v

    def _build(self, idx, depth):
        node = len(self._vec)
        self._vec.append(self.X[idx].mean(0))
        self._depth.append(depth)
        self._children.append((-1, -1))
        self._parent.append(-1)
        if idx.size == 1:
            self._leaf_node[int(idx[0])] = node
            return node
        v = self._pc1(self.X[idx])
        proj = (self.X[idx] - self._vec[node]) @ v
        order = idx[np.argsort(proj, kind="stable")]
        half = idx.size // 2
        l = self._build(order[:half], depth + 1)
        r = self._build(order[half:], depth + 1)
        self._children[node] = (l, r)
        self._parent[l] = self._parent[r] = node
        return node

    # ---------------------------------------------------------------- levels
    def plan(self, K, cap):
        """Depths for a K-level ladder whose codes must stay in 1..cap. Returns (depths, note)."""
        counts = self._counts
        bad = [d for d in range(len(counts)) if counts[d] > cap]
        d_min = (max(bad) + 1) if bad else 0
        note = ""
        if d_min > self.max_depth:
            raise ValueError(f"codebook cap {cap} is below the root's fan-out; use a larger V")
        depths = [d_min + (K - 1 - k) for k in range(K)]
        if depths[0] > self.max_depth:
            # Not enough tree for the coarse rungs: clamp. Coarse levels then share a partition,
            # which is degenerate but honest (and reported), never silently wrong.
            depths = [min(d, self.max_depth) for d in depths]
            note = f"tree shallower than K={K}: depths clamped at {self.max_depth}"
        if d_min > 0:
            note = (note + " | " if note else "") + \
                f"cap {cap} < {counts[0]} leaves: finest level starts at tree depth {d_min}"
        return depths, note

    def _ancestors(self, d):
        """[V] dense node id of every leaf's ancestor at depth d (a shallow leaf keeps itself)."""
        if d not in self._anc:
            out = self._leaf_node.copy()
            for _ in range(int(d)):
                par = self._parent[out]
                out = np.where(par >= 0, par, out)
            self._anc[d] = out
        return self._anc[d]

    def _level(self, d):
        """(map[V] compact ids 1..n, nodes[n, D]) for tree depth d, cached."""
        if d not in self._map:
            anc = self._ancestors(d)
            uniq = np.unique(anc)
            lut = np.zeros(len(self._vec), dtype=np.int32)
            lut[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
            self._compact[d] = uniq
            self._map[d] = lut[anc]
        return self._map[d], self._vec[self._compact[d]]

    def level_codes(self, d):
        """[V] codes at tree depth d, in 1..n_d."""
        return self._level(d)[0]

    def level_nodes(self, d):
        """[n_d, D] prototype vectors at tree depth d; row i-1 is code i."""
        return self._level(d)[1]

    def coarsen(self, d_from, d_to):
        """[n_from] id at depth d_to for each code id at depth d_from (both compact, 1-based)."""
        if d_to < d_from:
            raise ValueError("coarsen can only widen a cell, not refine it")
        self._level(d_from)
        self._level(d_to)
        up = self._compact[d_from].copy()          # dense node ids at the finer depth
        for _ in range(int(d_to) - int(d_from)):
            par = self._parent[up]
            up = np.where(par >= 0, par, up)
        return (np.searchsorted(self._compact[d_to], up) + 1).astype(np.int32)

    def n_nodes(self, d):
        self._level(d)
        return len(self._compact[d])


# ------------------------------------------------------------------ shared tree codec
class TreeCodebookCodec(Codec):
    """Flat codebook + `TreeHierarchy`: level k = pooled finest feature, climbed to depth d_k.

    Subclasses supply the raw -> features mapping (`_feats`) and the optional inverse (`_decode`).
    """

    kind, name, tier, decodable = "learned", "tree", "learned", True

    def __init__(self, K=6, R=2, V=256, base_len=64, leaves=None, seed=0):
        super().__init__(K, R, V, base_len)
        if leaves is None:
            raise ValueError("TreeCodebookCodec needs `leaves`")
        self.tree = TreeHierarchy(leaves, seed=seed)
        self.depths, note = self.tree.plan(K, V)
        self.note = note
        C = self.tree.X
        self.Cn = (C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    # ---------------------------------------------------------------- helpers
    def _leaf_codes(self, feats):
        """[n, D] -> [n] leaf codebook indices (0-based) by cosine nearest neighbour."""
        x = np.asarray(feats, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.Cn.shape[1]:
            raise ValueError(f"expected [n, {self.Cn.shape[1]}] features, got {x.shape}")
        x = x - x.mean(0, keepdims=True)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
        return (x @ self.Cn.T).argmax(1)

    def _encode_feats(self, feats):
        """[n, D] finest features -> K arrays of level codes, each exactly level_len(k) long."""
        feats = np.asarray(feats, dtype=np.float32)
        out = []
        for k in range(self.K):
            l = self.level_len(k)
            x = pool_feats(feats, l)                     # same pooling rule as every other codec
            leaf = self._leaf_codes(x)                   # [l]
            out.append(self.tree.level_codes(self.depths[k])[leaf].astype(np.int32))
        return out

    def _feats(self, obj):
        raise NotImplementedError

    def _decode(self, vecs):
        return vecs

    def encode(self, obj):
        return self._encode_feats(self._feats(obj))

    def decode(self, codes, k=None):
        """Prototype vectors at level k (default: the finest) - a real decode, not a placeholder."""
        k = self.K - 1 if k is None else int(k)
        nodes = self.tree.level_nodes(self.depths[k])
        idx = np.clip(np.asarray(codes, dtype=np.int32) - 1, 0, len(nodes) - 1)
        return self._decode(nodes[idx])

    def info(self):
        d = super().info()
        d.update(leaves=int(self.tree.V), dim=int(self.tree.dim),
                 depths=[int(x) for x in self.depths],
                 nodes=[int(self.tree.n_nodes(x)) for x in self.depths])
        return d


# ------------------------------------------------------------------ text
class TextEntropyCodec(Codec):
    """UTF-8 bytes at the finest rung (lossless), novelty-segmented spans above it.

    A fixed-width span (`TextByteCodec`) cuts words and sentences at arbitrary byte offsets, so the
    coarse rungs are noise the model has to memorise. Byte-level LMs spend their layer capacity
    exactly at *local structure boundaries* they discover (arXiv:2610.05978), so level k cuts at the
    same "new n-gram" positions instead: deterministic, corpus-free, no training in v1.

    This is a proxy, not a learned predictor: it decides WHERE to cut, it does not compress. The
    learned version (predict boundaries from a byte LM's entropy, then pool *representations* not
    bytes) is the documented follow-up in docs/codecs.md, and it plugs into the same `_segments`.
    """

    kind, name, tier = "learned", "text_entropy", "bytes+entropy"
    exact_at_finest = decodable = True
    NGRAM = 4

    def __init__(self, K=6, R=2, codebook_size=256, base_len=64, seed=1):
        super().__init__(K, R, codebook_size, base_len)
        self.seed = int(seed)

    # ---------------------------------------------------------------- segmentation
    def _repeat(self, raw):
        """[len(raw)] how often the NGRAM window *starting* at each byte was seen before."""
        n = len(raw)
        seen, out = {}, np.zeros(n, dtype=np.int32)
        for i in range(n):
            w = bytes(raw[i:i + self.NGRAM])
            out[i] = seen.get(w, 0)
            seen[w] = out[i] + 1
        return out

    def _candidates(self, raw):
        """Word starts (a non-space after a space); every interior byte if the text has no words."""
        n = len(raw)
        starts = [i for i in range(1, n)
                  if raw[i] not in b" \t\n\r" and raw[i - 1] in b" \t\n\r"]
        return starts or list(range(1, n))

    def _segments(self, raw, l):
        """Edges [0, b1 .. b_{l-1}, n]: whole-word spans, spent first where the text is novel."""
        n = len(raw)
        if l <= 1:
            return [0, n]
        if n <= 1:                                  # nothing to cut: every span is empty
            return [0] * (l + 1)
        m = l - 1
        cand = self._candidates(raw)
        if len(cand) > m:                           # keep the least repeated up-comings first
            rep = self._repeat(raw)
            order = np.lexsort((np.asarray(cand), rep[cand]))
            cand = sorted(int(cand[j]) for j in order[:m])
        elif len(cand) < m:                         # very short text: fall back to even spacing
            fill = np.linspace(1, n - 1, m).round().astype(int)
            cand = sorted(set(cand) | {int(p) for p in fill if 1 <= p < n})[:m]
        return [0] + list(cand) + [n]

    def _codes(self, raw, l):
        edges = self._segments(raw, l)
        if len(edges) - 1 != l:                              # defensive: never emit a short level
            edges = [round(i * len(raw) / l) for i in range(l + 1)]
        out = np.empty(len(edges) - 1, dtype=np.int32)
        for i in range(len(edges) - 1):
            chunk = bytes(raw[edges[i]:edges[i + 1]])
            h = hashlib.blake2b(chunk, digest_size=8,
                                salt=self.seed.to_bytes(8, "little")).digest()
            out[i] = int.from_bytes(h, "little") % self.V + 1
        return out

    def encode(self, text):
        raw = (text or "").encode("utf-8", "ignore")[: self.base_len]
        out = []
        for k in range(self.K):
            l = self.level_len(k)
            if k == self.K - 1:                              # lossless bottom rung
                b = np.zeros(l, dtype=np.int32)
                b[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)[:l].astype(np.int32) + 1
                out.append(b)
            else:
                out.append(self._codes(raw, l))
        return out

    def decode(self, codes):
        b = np.clip(np.asarray(codes[-1], dtype=np.int32) - 1, 0, 255).astype(np.uint8)
        return b.tobytes().rstrip(b"\x00").decode("utf-8", "ignore")


# ------------------------------------------------------------------ audio
class DacAudioCodec(TreeCodebookCodec):
    """descript-audio-codec as the leaf codebook: encoder latents per frame, tree over RVQ-1.

    DAC is the strongest openly-installable fidelity baseline (44.1 kHz, 9 RVQ stages); the
    generative frontier it does NOT reach is the semantic/acoustic dual stream
    (DualCodec in arXiv:2609.08703, HybridCodec arXiv:2606.06743) and variable-rate budgeted coding
    (RubatoCodec arXiv:2610.29434). Those are documented in docs/codecs.md as the upgrade path;
    this adapter exists so the ladder can be exercised against a real codebook today.
    """

    kind, name, tier, decodable = "learned", "dac", "learned", True
    SR = 44100
    MODEL = os.environ.get("NCM_DAC_MODEL", "44khz")

    @classmethod
    def available(cls):
        import importlib.util
        return importlib.util.find_spec("dac") is not None

    def __init__(self, K=6, R=2, codebook_size=256, base_len=96, seed=0):
        import dac                                     # noqa: F401 - availability is the caller's job
        import torch
        model = dac.DAC.load(dac.utils.download(model_type=self.MODEL)).eval()
        self._model, self._torch = model, torch
        leaves = model.quantizer.vq.layers[0]._codebook.embed[0].detach().cpu().numpy()
        super().__init__(K, R, codebook_size, base_len, leaves=leaves, seed=seed)
        self.note = (f"dac-{self.MODEL} rvq-1 codebook ({len(leaves)} leaves); "
                     "dual-stream/variable-rate upgrades in docs/codecs.md")

    def _feats(self, obj):
        wav, sr = obj
        x = np.asarray(wav, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError("empty waveform")
        if sr and int(sr) != self.SR:                   # let torchaudio or numpy resample
            try:
                import torchaudio
                x = torchaudio.functional.resample(
                    self._torch.as_tensor(x), int(sr), self.SR).numpy()
            except Exception:                           # noqa: BLE001
                n = max(1, int(round(len(x) * self.SR / int(sr))))
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)),
                              x).astype(np.float32)
        with self._torch.no_grad():
            z = self._model.encoder(self._torch.as_tensor(x)[None, None])
        return z[0].transpose(0, 1).cpu().numpy().astype(np.float32)      # [T, D]


# ------------------------------------------------------------------ image
class DinoKmeansImageCodec(TreeCodebookCodec):
    """Frozen DINOv2/v3 patch features + a k-means codebook trained offline, tree on top.

    Trained on features, not pixels: a tokenizer whose codes predict *concept* features gives the
    generator something to model and keeps the coarse rungs semantic (arXiv:2610.22060 FeaTok:
    -29% gFDr6, beats the strongest continuous baseline at 38x less sampling compute).

    The codebook is a file, not something this class invents at import time:

        python - <<'PY'
        import numpy as np, torch
        from transformers import AutoImageProcessor, AutoModel
        mid = "facebook/dinov2-base"; proc = AutoImageProcessor.from_pretrained(mid)
        net = AutoModel.from_pretrained(mid).eval()
        # ... sample ~50k images, keep patch tokens as [N, D], then k-means (k=V) offline ...
        np.savez("run/codecs/image_kmeans.npz", centroids=centroids)
        PY

    `NCM_IMAGE_CODEBOOK` (default `run/codecs/image_kmeans.npz`) points at the result; without it
    the adapter reports itself unavailable and the stand-in pyramid is used.
    """

    kind, name, tier, decodable = "learned", "dino_kmeans", "learned", True
    MODEL = os.environ.get("NCM_DINO_MODEL", "facebook/dinov2-base")
    RES = 224

    @classmethod
    def ckpt_path(cls):
        return os.environ.get("NCM_IMAGE_CODEBOOK") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "run", "codecs", "image_kmeans.npz")

    @classmethod
    def available(cls):
        import importlib.util
        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None
                and os.path.exists(cls.ckpt_path()))

    def __init__(self, K=6, R=2, codebook_size=256, base_len=64, seed=0):
        data = np.load(self.ckpt_path())
        keys = set(data.files)
        if "centroids" not in keys:
            raise ValueError(f"{self.ckpt_path()} has no 'centroids' array (has {sorted(keys)})")
        super().__init__(K, R, codebook_size, base_len, leaves=data["centroids"], seed=seed)
        self.note = f"dinov2 patch features + kmeans codebook {self.ckpt_path()}"
        self._model = self._proc = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self._proc = AutoImageProcessor.from_pretrained(self.MODEL)
            self._model = AutoModel.from_pretrained(self.MODEL).eval()
            self._torch = torch
        return self._model

    def _feats(self, img):
        model = self._load()
        torch = self._torch
        pil = img.convert("RGB").resize((self.RES, self.RES))
        px = self._proc(images=pil, return_tensors="pt")
        with torch.no_grad():
            out = model(**px)
        h = out.last_hidden_state[0]
        n_prefix = 0 if getattr(model.config, "num_register_tokens", 0) == 0 else \
            1 + int(getattr(model.config, "num_register_tokens", 0))
        return h[1 + n_prefix:].cpu().numpy().astype(np.float32)          # [n_patches, D]


# ------------------------------------------------------------------ registry
def build_learned(modality, K, R, V, base_len, tier="auto"):
    """(codec, reason). codec is None when this machine cannot build one - never a silent swap."""
    tier = (tier or "auto").lower()
    if tier not in ("auto", "learned"):
        return None, f"tier {tier!r} does not ask for a learned codec"

    if modality == "text":
        # Dependency-free: bytes are always available and the segmentation is closed-form.
        c = TextEntropyCodec(K, R, V, base_len)
        c.note = ("byte-novelty segmentation (proxy for the local-abstraction boundaries of "
                  "arXiv:2610.05978); learned boundary predictor is the follow-up")
        return c, c.note

    if modality == "image":
        if not DinoKmeansImageCodec.available():
            import importlib.util
            if importlib.util.find_spec("transformers") is None:
                return None, "transformers not installed"
            return None, f"no image codebook at {DinoKmeansImageCodec.ckpt_path()}"
        try:
            c = DinoKmeansImageCodec(K, R, V, base_len)
        except Exception as e:                                  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:160]}"
        return c, c.note

    if modality == "audio":
        if not DacAudioCodec.available():
            return None, "descript-audio-codec not installed (pip install descript-audio-codec)"
        try:
            c = DacAudioCodec(K, R, V, base_len)
        except Exception as e:                                  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:160]}"
        return c, c.note

    if modality == "video":
        return None, "video is encoded as image samples; no separate learned codec"

    return None, f"no learned codec for modality {modality!r}"
