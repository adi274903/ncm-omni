"""Omni generator: one transformer over level-major packs of every modality.

Pack layout (see schedule.py):   [L0: text|image|audio|video][L1: ...] ... [LK-1: ...]

  block-diagonal mask  -> level k sees its own level across ALL modalities (the omni claim),
                          and nothing finer. One forward predicts every code of every modality
                          at that altitude.
  mask_mode="ar"       -> same pack, block-causal + causal within the level: a matched-params
                          autoregressive baseline over the raster order, one token at a time.

Conditioning follows the paper in spirit, not in letter: NCM conditions level k on the codec's
reconstruction z_hat(k-1); our codecs are discrete-only, so we condition on the mean embedding
of the level-(k-1) codes actually fed in. Named deviation, `cond_mode="gt_embed"`.
"""
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

try:
    from core import Block, HAS_FLEX, RMSNorm, precompute_rope_cos_sin
except ModuleNotFoundError:                                # core.py lives in the sibling package
    for _cand in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  os.path.expanduser("~/projects/ncm_fused"),
                  os.environ.get("NCM_CORE_DIR", "")):
        if _cand and os.path.exists(os.path.join(_cand, "core.py")):
            sys.path.insert(0, _cand)
            break
    from core import Block, HAS_FLEX, RMSNorm, precompute_rope_cos_sin
from omni.schedule import MODALITIES, OmniSchedule

try:
    from torch.nn.attention.flex_attention import create_block_mask
except Exception:                                          # pragma: no cover
    create_block_mask = None


try:                                # per-level widths differ (24 vs 56 keys), so flex recompiles
    torch._dynamo.config.cache_size_limit = 32   # once per level and hits the default limit of 8
except Exception:                   # -> silent fallback to the unfused eager kernel
    pass


def normalize_vocab(vocab, modalities, K):
    """-> {m: (V_0, ..., V_{K-1})}.

    Accepts, in order of age:
      * {m: int}                       one codebook per modality, every level (the old layout)
      * {m: [int]*K} or {m: {k: int}}  a width per level (heterogeneous head: the finest text
                                       level holds BPE ids, the coarse levels hold 256-way codes)
      * int                            the same for every modality

    Keeping every level the same width is still the default and still the only layout that loads
    the pre-existing checkpoints: a heterogeneous model cannot read a `heads.text.weight`.
    """
    out = {}
    for m in modalities:
        v = vocab.get(m, 256) if isinstance(vocab, dict) else vocab
        if isinstance(v, dict):
            seq = [int(v[k]) for k in range(K)]
        elif isinstance(v, (list, tuple)):
            seq = [int(x) for x in v]
            seq = (seq + [seq[-1]] * K)[:K]
        else:
            seq = [int(v)] * K
        assert all(x >= 1 for x in seq), (m, seq)
        out[m] = tuple(seq)
    return out


class OmniConfig:
    def __init__(self, sched: OmniSchedule, vocab: Dict[str, int], n_embd=384, n_layer=6,
                 n_head=6, n_kv_head=2, dropout=0.0, qk_norm=True,
                 cond_mode="gt_embed", mask_mode="block", max_pos=None,
                 n_experts=0, expert_map=None, moe_ffn=0, use_flex=None):
        self.sched = sched
        self.vocab_raw = vocab
        self.vocab = normalize_vocab(vocab, sched.modalities, sched.K)
        self.n_embd, self.n_layer = n_embd, n_layer
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.dropout, self.qk_norm = dropout, qk_norm
        self.cond_mode, self.mask_mode = cond_mode, mask_mode
        self.n_experts = int(n_experts or 0)
        self.expert_map = expert_map
        self.moe_ffn = int(moe_ffn or 0)          # 0 = derive the expert width from mlp_ratio
        # None = auto (flex on sm_80+), False = force the dense SDPA mask. Measured on an A100 at
        # T=1764, B=8, H=16: dense materialises [B,H,T,T] = 1.59 GiB per tensor and its backward
        # keeps several, so it OOMs while asking for 1.48 GiB with 33.6 GiB already resident - and
        # it runs at 28 TFLOPS against 99 for the block-sparse path. Flex's Triton kernel failing to
        # compile (182 KB of shared memory wanted, 166 KB available) is still the better trade: the
        # unfused fallback is fast. Leave this None unless the packed length is small.
        self.use_flex = use_flex
        self.max_pos = max_pos or max(sched.base_len[m] for m in sched.modalities)

    # -- vocab -------------------------------------------------------------
    def vocab_at(self, m, k):
        return self.vocab[m][k]

    def vocab_max(self, m):
        return max(self.vocab[m])

    def is_hetero(self, m=None):
        mods = self.sched.modalities if m is None else (m,)
        return any(len(set(self.vocab[mm])) > 1 for mm in mods)

    def to_dict(self):
        # A uniform modality serialises back to the old `int` form so old cfgs round-trip.
        v = {m: (list(seq) if len(set(seq)) > 1 else seq[0]) for m, seq in self.vocab.items()}
        return dict(K=self.sched.K, R=self.sched.R, base_len=dict(self.sched.base_len),
                    vocab=v, n_embd=self.n_embd, n_layer=self.n_layer,
                    n_head=self.n_head, n_kv_head=self.n_kv_head, dropout=self.dropout,
                    qk_norm=self.qk_norm, cond_mode=self.cond_mode, mask_mode=self.mask_mode,
                    n_experts=self.n_experts, expert_map=self.expert_map)


def level_ids_for(sched: OmniSchedule, T: int, device="cpu"):
    """Level id per packed position, truncated to a prefix of length T (generation grows T)."""
    lev, mod, pos = sched.positions()
    return lev[:T].to(device), mod[:T].to(device), pos[:T].to(device)


_CACHE_MAX = 8                   # entries for _STATIC below


def _geometry(sched: OmniSchedule):
    """Everything a per-T tensor depends on, as a hashable key.

    Not `id(sched)`: a schedule rebuilt from the same cfg between calls must hit the same entry.
    """
    return (tuple(sorted(sched.base_len.items())), sched.K, tuple(sorted(sched.Rm.items())),
            sched.prefix_len, tuple(sched.modalities), sched.null_code)


_STATIC = {}                     # (geometry, T, device, head_dim, emap) -> lev, mod, pos, rope, eid


def static_tensors(sched: OmniSchedule, T: int, device, head_dim: int, emap=None):
    """Level ids, positional ids, RoPE tables and expert routing for a packed length T.

    Built once instead of once per step. encode() used to rebuild all of it on every call: three
    fresh tensors and their host->device copies, a python loop over the modality masks, and two
    sin/cos tables over T positions. They are pure functions of the schedule and T, so at 0.18 s
    per step that was pure launch overhead that never touched the maths.
    """
    key = (_geometry(sched), int(T), str(device), int(head_dim),
           None if emap is None else tuple(emap))
    hit = _STATIC.get(key)
    if hit is None:
        lev, mod, pos = level_ids_for(sched, T, device)
        eid = None
        if emap is not None:
            eid = torch.zeros_like(mod)
            for i, ex in enumerate(emap):        # prefix slots (-1) stay on expert 0
                eid[mod == i] = ex
        # per-expert position index, so the MoE path never branches on tensor values
        plan = None
        if eid is not None and emap is not None:
            plan = tuple((eid == e).nonzero(as_tuple=False).squeeze(1)
                         for e in range(max(emap) + 1))
        hit = (lev, mod, pos, precompute_rope_cos_sin(pos, head_dim), eid, plan)
        if len(_STATIC) >= _CACHE_MAX:
            _STATIC.pop(next(iter(_STATIC)))
        _STATIC[key] = hit
    return hit


def trunc_mask(sched: OmniSchedule, T: int, dtype=torch.float32, device="cpu",
               cross_level=False, mask_mode="block"):
    """Additive [1,1,T,T] mask for a packed length T.

    Deliberately not cached. It was, keyed on (geometry, T, dtype, device, cross_level, mode), and
    the cache had to refuse anything over 64 MB - i.e. it silently rebuilt every step at any
    T above ~4000, which is exactly the range we care about. So it bought nothing where it
    mattered and was a second place for the mask to be wrong. Build it; it is pure.
    """
    lev, _, _ = level_ids_for(sched, T, device)
    q, kv = lev[:, None], lev[None, :]
    if mask_mode == "ar":
        # raster order: everything strictly before, plus the prefix
        idx = torch.arange(T, device=device)
        allowed = idx[None, :] <= idx[:, None]
    elif mask_mode == "strict":
        # Generation reality: every code of level k is emitted in ONE forward pass, so no position
        # may see a sibling of its own level. Level k attends to levels < k only. Level 0 has no
        # coarser level, so it attends to its own block - which pack() always fills with NULL, at
        # train time too, so there is nothing to copy there either.
        allowed = (kv == -1) | ((q == 0) & (kv == 0)) | ((kv >= 0) & (kv < q))
    else:
        allowed = (kv == -1) | (kv == q)
        if cross_level:
            allowed = allowed | ((q >= 0) & (kv >= 0) & (kv < q))
    m = torch.zeros(T, T, dtype=dtype, device=device)
    m.masked_fill_(~allowed, float("-inf"))
    return m.view(1, 1, T, T)


def kv_ids(lev):                                            # kept for clarity in trunc_mask
    return lev


_FLEX_CACHE = OrderedDict()      # LRU: the block mask is ~200 MB at T=7119, so keep few
_FLEX_CACHE_MAX = 8


def trunc_flex(sched: OmniSchedule, B: int, H: int, T: int, device="cuda",
               cross_level=False, mask_mode="block"):
    """Block mask for the fused kernel, built once per (length, mask) and broadcast.

    Two things matter at kill-test length. The mask depends only on (level, position), never on
    b or h, so B/H go in as None: create_block_mask then derives block sparsity from one T x T
    grid instead of a B x H x T x T one, which at T=7119, B=2, H=12 is 9 GiB of floats and OOMs
    on any consumer card. And it is cached, because rebuilding a 200 MB mask every step costs
    more than the forward pass at these sizes.
    """
    assert create_block_mask is not None, "flex_attention unavailable"
    key = (T, mask_mode, bool(cross_level), str(device),
           tuple(sorted(sched.base_len.items())), sched.K)
    hit = _FLEX_CACHE.get(key)
    if hit is not None:
        _FLEX_CACHE.move_to_end(key)
        return hit
    lev, _, _ = level_ids_for(sched, T, device)

    def mask_mod(b, h, q_idx, kv_idx):
        if mask_mode == "ar":
            return kv_idx <= q_idx
        ql, kl = lev[q_idx], lev[kv_idx]
        if mask_mode == "strict":
            return (kl == -1) | ((ql == 0) & (kl == 0)) | ((kl >= 0) & (kl < ql))
        allowed = (kl == -1) | (kl == ql)
        if cross_level:
            allowed = allowed | ((ql >= 0) & (kl >= 0) & (kl < ql))
        return allowed

    bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device)
    _FLEX_CACHE[key] = bm
    while len(_FLEX_CACHE) > _FLEX_CACHE_MAX:
        _FLEX_CACHE.popitem(last=False)
    return bm


class OmniNCM(nn.Module):
    def __init__(self, cfg: OmniConfig):
        super().__init__()
        self.cfg, self.sched = cfg, cfg.sched
        d, K = cfg.n_embd, cfg.sched.K
        # The embedding table is sized by the widest level of that modality; coarse levels simply
        # never index past their own (smaller) range.
        self.tok_emb = nn.ModuleDict({m: nn.Embedding(cfg.vocab_max(m) + 1, d)
                                      for m in cfg.sched.modalities})
        self.mod_emb = nn.Embedding(len(cfg.sched.modalities) + 1, d)   # +1 = prefix slot
        self.level_emb = nn.Embedding(K, d)
        self.pos_emb = nn.Embedding(cfg.max_pos + 1, d)
        self.cond_proj = nn.Linear(d, d)
        # MoE: shared trunk, per-modality experts. A uniform model keeps the dense FFN, so old
        # checkpoints keep loading.
        self.moe = bool(cfg.n_experts)
        if self.moe:
            from omni.moe import MoEBlock, resolve_expert_map
            self._emap, self.n_experts = resolve_expert_map(
                cfg.sched.modalities, cfg.expert_map, cfg.n_experts)
            blocks = [MoEBlock(Block(d, cfg.n_head, cfg.n_kv_head, cfg.dropout, qk_norm=cfg.qk_norm),
                               d, self.n_experts, ffn_hidden=cfg.moe_ffn or None)
                      for _ in range(cfg.n_layer)]
        else:
            self._emap, self.n_experts = None, 0
            blocks = [Block(d, cfg.n_head, cfg.n_kv_head, cfg.dropout, qk_norm=cfg.qk_norm)
                      for _ in range(cfg.n_layer)]
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = RMSNorm(d)
        # One head per modality when every level shares a width (old layout, loads old ckpts);
        # one head per (modality, level) when the finest level is wider (BPE ids).
        self.heads = nn.ModuleDict()
        for m in cfg.sched.modalities:
            if cfg.is_hetero(m):
                for k in range(K):
                    self.heads[f"{m}_L{k}"] = nn.Linear(d, cfg.vocab_at(m, k), bias=False)
            else:
                self.heads[m] = nn.Linear(d, cfg.vocab_at(m, 0), bias=False)

    def head(self, m, k):
        key = f"{m}_L{k}"
        return self.heads[key] if key in self.heads else self.heads[m]

    # ------------------------------------------------------------------ packing
    def _mod_level_emb(self, m, k, codes_mk, cond):
        l = codes_mk.size(-1)
        e = (self.tok_emb[m](codes_mk)
             + self.level_emb.weight[k]
             + self.mod_emb.weight[self.sched.modalities.index(m)]
             + self.pos_emb(torch.arange(l, device=codes_mk.device)))
        return e if cond is None else e + self.cond_proj(cond)

    def pack(self, codes: Dict[str, torch.Tensor], upto: Optional[int] = None,
             dummy_level: Optional[int] = None):
        """codes[m] = [B, flat_len(m)] -> packed [B, T, d] over levels < upto.

        The packed input at a position is NEVER that position's own target code, in either mode:

          * `strict` (level-parallel): level k is fed the codes of level k-1, upsampled to l_k.
            Level 0 is fed NULL - it has no coarser level, so it is a prior over the coarsest
            codes and nothing else.
          * `ar`: the raster-shifted sequence, i.e. position j is fed code j-1.

        Without that shift the head can read its own input token straight out of the residual
        stream (the token embedding is in `h` by construction), so the model learns an identity
        map: train loss -> 1e-8, and at generation - where the input is NULL or a guess - it
        emits garbage, scoring BELOW a constant predictor. `block` keeps the old (leaky) layout
        for comparison and should not be quoted.
        """
        S, K = self.sched, self.sched.K
        B = next(iter(codes.values())).size(0)
        stop = K if upto is None else upto
        mode = self.cfg.mask_mode
        shifted = {}
        if mode == "ar":                                  # position j is fed code j-1
            for m in S.modalities:
                c = codes[m]
                shifted[m] = torch.cat([torch.zeros(B, 1, dtype=c.dtype, device=c.device),
                                        c[:, :-1]], -1)
        parts, conds = [], {}
        for k in range(spread(stop, dummy_level)):
            block = []
            for m in S.modalities:
                l = S.level_len(m, k)
                off = S.offsets(m)[k]
                if mode == "strict":
                    if k == 0:
                        c = torch.zeros(B, l, dtype=torch.long, device=self._dev)
                    else:
                        prev = codes[m][:, S.offsets(m)[k - 1]:S.offsets(m)[k]]
                        lp = prev.size(-1)
                        idx = (torch.arange(l, device=prev.device) * lp) // l   # nearest neighbour
                        c = prev[:, idx]
                    if dummy_level is not None and k > dummy_level:
                        c = torch.zeros(B, l, dtype=torch.long, device=self._dev)
                elif mode == "ar":
                    c = shifted[m][:, off:off + l]
                elif dummy_level is not None and k == dummy_level:
                    c = torch.zeros(B, l, dtype=torch.long, device=self._dev)
                else:
                    c = codes[m][:, off:off + l]
                block.append(self._mod_level_emb(m, k, c, conds.get(m)))
            parts.append(torch.cat(block, 1))
            # condition for the next level = mean of this level's embeddings, per modality
            for i, m in enumerate(S.modalities):
                conds[m] = block[i].mean(1, keepdim=True).expand(-1, S.level_len(m, k + 1), -1) \
                    if k + 1 < K else None
        return torch.cat(parts, 1)

    @property
    def _dev(self):
        return self.pos_emb.weight.device

    # ------------------------------------------------------------------ forward
    def _expert_ids(self, mod_idx):
        """Packed-position modality index -> expert id under this model's expert map.

        preflight calls this to count the tokens each expert would receive, so a forced mapping that
        leaves an expert with nothing shows up before training instead of being inferred from a flat
        loss curve. It was referenced from omni/preflight.py long before it existed here: the MoE
        branch of preflight had never run, so nothing caught the missing method.
        """
        if self._emap is None:
            raise ValueError("dense model has no experts; _expert_ids needs an expert_map")
        eid = torch.zeros_like(mod_idx)
        for i, ex in enumerate(self._emap):        # prefix slots keep expert 0, as in static_tensors
            eid[mod_idx == i] = ex
        return eid

    def encode(self, x, mask_mode=None, cross_level=False, use_flex=None):
        T, B = x.size(1), x.size(0)
        mask_mode = mask_mode or self.cfg.mask_mode
        if use_flex is None:
            use_flex = self.cfg.use_flex      # config override; False forces the dense SDPA mask
        if use_flex is None:
            # flex_attention's fused kernel needs sm_80+. On a free T4 (sm_75) it either errors or
            # falls back to something slower than the dense SDPA path, so gate on capability.
            use_flex = (HAS_FLEX and x.is_cuda
                        and torch.cuda.get_device_capability(x.device)[0] >= 8)
        kw = {}
        if use_flex:
            kw["block_mask"] = trunc_flex(self.sched, B, self.cfg.n_head, T, x.device,
                                          cross_level, mask_mode)
        else:
            kw["dense_mask"] = trunc_mask(self.sched, T, x.dtype, x.device, cross_level, mask_mode)
        lev, mod, pos, rope, eid, plan = static_tensors(
            self.sched, T, x.device, self.cfg.n_embd // self.cfg.n_head,
            tuple(self._emap) if self._emap is not None else None)
        h = x
        for b in self.blocks:
            h = b(h, rope=rope, mod_id=eid, plan=plan, **kw) if self.moe \
                else b(h, rope=rope, **kw)
        return self.ln_f(h)


    def forward(self, batch, cross_level=False, use_flex=None):
        """batch = {"codes": {m: [B,flat]}, "present": {m: [B] bool}} -> {(m,k): logits}."""
        codes = batch["codes"]
        x = self.pack(codes)
        h = self.encode(x, cross_level=cross_level, use_flex=use_flex)
        out = {}
        for k in range(self.sched.K):
            q0 = self._qoff(k)
            for m in self.sched.modalities:                        # modality slot inside the level block
                l = self.sched.level_len(m, k)
                off_in_block = sum(self.sched.level_len(mm, k)
                                   for mm in self.sched.modalities[:self.sched.modalities.index(m)])
                out[(m, k)] = self.head(m, k)(h[:, q0 + off_in_block:q0 + off_in_block + l])
        return out

    def _qoff(self, k):
        """Packed offset of level k's *text* slot plus preceding modalities."""
        S = self.sched
        return sum(S.block_width(j) for j in range(k))

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, batch, level_cutoff=None, temperature=0.0, use_flex=None):
        """Coarse-to-fine: NFE == level_cutoff (one forward per level). Returns codes[m] flat."""
        S = self.sched
        K = S.K if level_cutoff is None else level_cutoff
        codes = {m: torch.zeros_like(batch["codes"][m]) for m in S.modalities}
        for k in range(K):
            x = self.pack(codes, upto=k + 1, dummy_level=k)
            h = self.encode(x, use_flex=use_flex)
            q0 = self._qoff(k)
            c = 0
            for m in S.modalities:
                l = S.level_len(m, k)
                lg = self.head(m, k)(h[:, q0 + c:q0 + c + l])
                c += l
                if temperature and temperature > 0:
                    idx = torch.multinomial(F.softmax(lg / temperature, -1).reshape(-1, lg.size(-1)), 1)
                    idx = idx.reshape(lg.size(0), l)
                else:
                    idx = lg.argmax(-1)
                off = S.offsets(m)[k]
                codes[m][:, off:off + l] = idx
        return codes

    @torch.no_grad()
    def generate_ar(self, batch, max_len=None, use_flex=None):
        """Autoregressive baseline: one forward per token. NFE == number of tokens."""
        S = self.sched
        dev = self._dev
        flat = {m: torch.zeros(S.flat_len(m), dtype=torch.long, device=dev) for m in S.modalities}
        order = [(k, m, i) for k in range(S.K) for m in S.modalities
                 for i in range(S.level_len(m, k))]
        if max_len:
            order = order[:max_len]
        buf = {m: flat[m].unsqueeze(0) for m in S.modalities}
        for (k, m, i) in order:
            x = self.pack(buf, upto=k + 1, dummy_level=k)
            h = self.encode(x, use_flex=use_flex)
            q0 = self._qoff(k) + sum(S.level_len(mm, k) for mm in S.modalities[:S.modalities.index(m)]) + i
            idx = self.head(m, k)(h[:, q0:q0 + 1]).argmax(-1)
            buf[m][0, S.offsets(m)[k] + i] = idx[0, 0]
        return {m: buf[m] for m in S.modalities}, len(order)


def spread(stop, dummy_level):
    """How many levels to build: up to `stop`, extended by one if the dummy sits at the end."""
    return stop if dummy_level is None else max(stop, dummy_level + 1)


# ---------------------------------------------------------------------- loss
def omni_loss(logits, batch, sched: OmniSchedule, label_smoothing=0.0, chunk=None):
    """CE over every present, non-null code. Absent modalities and padded slots are masked.

    Sync-free on purpose: the divisor is accumulated as a tensor and an all-masked level is
    zeroed with torch.where instead of `if not keep.any(): continue`. Each of those old branches
    was a device-wide barrier in the middle of the graph - 18 of them per step at K=6, three
    modalities - so the forward could never be queued ahead of the host. The sum is identical.
    """
    codes, pres = batch["codes"], batch["present"]
    total, n = None, None
    for k in range(sched.K):
        for m in sched.modalities:
            l, off = sched.level_len(m, k), sched.offsets(m)[k]
            tg = codes[m][:, off:off + l]
            keep = (tg != sched.null_code) & pres[m].unsqueeze(1)
            cnt = keep.sum()
            lg = logits[(m, k)]
            flat_lg = lg.reshape(-1, lg.size(-1))
            flat_keep = keep.reshape(-1)
            # one masked target vector for both paths: -100 is CE's ignore index, null_code is a
            # legal code and must never be counted as a target.
            flat_tg = tg.reshape(-1).clone()
            flat_tg[~flat_keep] = -100
            if chunk:
                loss_k = flat_lg.new_zeros(())
                for i in range(0, flat_lg.size(0), chunk):
                    loss_k = loss_k + F.cross_entropy(
                        flat_lg[i:i + chunk], flat_tg[i:i + chunk], reduction="sum",
                        ignore_index=-100, label_smoothing=label_smoothing)
            else:
                loss_k = F.cross_entropy(flat_lg, flat_tg, reduction="sum",
                                         ignore_index=-100, label_smoothing=label_smoothing)
            # every target masked -> CE is nan; contribute nothing instead of poisoning the mean
            loss_k = torch.where(cnt > 0, loss_k, loss_k.new_zeros(()))
            total = loss_k if total is None else total + loss_k
            n = cnt if n is None else n + cnt
    if total is None or n is None:
        return torch.zeros((), device=self_dev(codes))
    return total / n.clamp(min=1)


def omni_loss_stats(logits, batch, sched: OmniSchedule, label_smoothing=0.0):
    """-> (mean CE, {(m,k): {"ce": float, "n": int}}).

    Same arithmetic as `omni_loss`, but it also returns the per-level breakdown, which is what
    the preflight gates on and what TensorBoard plots. Kept separate so the optimiser path stays
    the single scalar it always was.
    """
    codes, pres = batch["codes"], batch["present"]
    total, n, out = None, 0, {}
    for k in range(sched.K):
        for m in sched.modalities:
            l, off = sched.level_len(m, k), sched.offsets(m)[k]
            tg = codes[m][:, off:off + l]
            keep = (tg != sched.null_code) & pres[m].unsqueeze(1)
            if not keep.any():
                continue
            lg = logits[(m, k)]
            flat_lg, flat_keep = lg.reshape(-1, lg.size(-1)), keep.reshape(-1)
            flat_tg = tg.reshape(-1).clone()
            flat_tg[~flat_keep] = -100
            loss_k = F.cross_entropy(flat_lg, flat_tg, reduction="sum", ignore_index=-100,
                                     label_smoothing=label_smoothing)
            cnt = int(flat_keep.sum())
            out[(m, k)] = {"ce": float(loss_k.detach()) / max(1, cnt), "n": cnt}
            total = loss_k if total is None else total + loss_k
            n += cnt
    mean = (total / max(1, n)) if total is not None else torch.zeros((), device=self_dev(codes))
    return mean, out


def self_dev(codes):
    return next(iter(codes.values())).device


# ---------------------------------------------------------------------- selfcheck
def selfcheck(verbose=True):
    """Invariants that must hold before any training. Run this on the target GPU, not in review."""
    from omni.schedule import OmniSchedule as Sch

    sched = Sch(base_len={"text": 8, "image": 8, "audio": 8, "video": 4}, K=3, R=2)
    vocab = {m: 16 for m in MODALITIES}
    cfg = OmniConfig(sched, vocab, n_embd=32, n_layer=2, n_head=4, n_kv_head=2)
    net = OmniNCM(cfg)
    B = 2
    codes = {m: torch.randint(1, 16, (B, sched.flat_len(m))) for m in MODALITIES}
    pres = {m: torch.ones(B, dtype=torch.bool) for m in MODALITIES}
    batch = {"codes": codes, "present": pres}
    out = net(batch)

    # 1. shapes
    for m in MODALITIES:
        for k in range(sched.K):
            assert out[(m, k)].shape == (B, sched.level_len(m, k), vocab[m]), (m, k, out[(m, k)].shape)

    # 1b. expert width. `moe_ffn` has to mean what it says: the mlp_ratio path rounds through
    # int(2*r*d/3//8*8), so "360M per expert" comes out a slightly different number, and the whole
    # 50x token budget is derived from this count. Asserted, not trusted.
    wcfg = OmniConfig(sched, vocab, n_embd=32, n_layer=2, n_head=4, n_kv_head=2,
                      n_experts=3, moe_ffn=128)
    wnet = OmniNCM(wcfg)
    e0 = wnet.blocks[0].experts
    per_layer = sum(w.numel() for w in (e0.w_gate[0].weight, e0.w_up[0].weight,
                                        e0.w_down[0].weight))
    assert per_layer == 3 * wcfg.n_embd * 128, {
        "h": e0.w_gate[0].weight.shape[0], "per_layer": per_layer, "want": 3 * wcfg.n_embd * 128}
    assert e0.w_gate[0].weight.shape[0] == 128, "ffn_hidden not applied to the gate"
    got = sum(p.numel() for n, p in wnet.named_parameters() if ".experts.w_" in n)
    want = wcfg.n_layer * 3 * (3 * wcfg.n_embd * 128 + 2 * 128 + wcfg.n_embd)  # + gate/up/down bias
    assert got == want, f"expert total: {got} != {want}"

    # 2. level k cannot see level k+1: perturb the finest level, the coarsest logits must not move
    net.eval()
    with torch.no_grad():
        base = net(batch)
        c2 = {m: codes[m].clone() for m in MODALITIES}
        fine_off = sched.offsets("text")[sched.K - 1]
        c2["text"][:, fine_off:] = (c2["text"][:, fine_off:] + 1) % 16
        after = net({"codes": c2, "present": pres})
        d0 = (base[("text", 0)] - after[("text", 0)]).abs().max().item()
        assert d0 == 0.0, f"level 0 moved when the finest level changed: {d0}"

    # 3. cross-level knob is wired: turning it on MUST change level-1 logits
    with torch.no_grad():
        cl = net(batch, cross_level=True)
        d1 = (base[("text", 1)] - cl[("text", 1)]).abs().max().item()
        assert d1 > 0, "cross_level=True changed nothing - the mask knob is dead"

    # 4. gradients reach every modality and every level
    net.train()
    loss = omni_loss(net(batch), batch, sched)
    loss.backward()
    for m in MODALITIES:
        g = net.tok_emb[m].weight.grad
        assert g is not None and g.abs().sum() > 0, f"no gradient on {m} embeddings"
    for k in range(sched.K):
        assert net.level_emb.weight.grad[k].abs().sum() > 0, f"level {k} unused"

    # 5. absent modality is masked, not silently trained
    p2 = {m: torch.ones(B, dtype=torch.bool) for m in MODALITIES}
    p2["audio"] = torch.zeros(B, dtype=torch.bool)
    l_absent = omni_loss(net(batch), {"codes": codes, "present": p2}, sched)
    l_full = omni_loss(net(batch), batch, sched)
    assert l_absent.item() != l_full.item(), "masking an absent modality did not change the loss"

    # 6. AR mode is causal: perturbing the last position cannot change earlier logits
    net.eval()
    cfg_ar = OmniConfig(sched, vocab, n_embd=32, n_layer=2, n_head=4, n_kv_head=2, mask_mode="ar")
    ar = OmniNCM(cfg_ar).eval()
    with torch.no_grad():
        a0 = ar(batch)[("text", 0)]
        c3 = {m: codes[m].clone() for m in MODALITIES}
        c3["text"][:, -1:] = (c3["text"][:, -1:] + 3) % 16
        a1 = ar({"codes": c3, "present": pres})[("text", 0)]
        assert (a0[:, :-1] - a1[:, :-1]).abs().max().item() == 0.0, "AR mask leaks the future"

    # 7. generation costs exactly K forwards and returns the right shapes
    gen = net.generate(batch)
    for m in MODALITIES:
        assert gen[m].shape == (B, sched.flat_len(m)), (m, gen[m].shape)

    if verbose:
        print("omni selfcheck: 7/7 invariants hold")
    return True


if __name__ == "__main__":
    selfcheck()
