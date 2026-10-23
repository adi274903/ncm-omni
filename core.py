"""NCM (arXiv:2610.14162) level-parallel generator - kernel-optimised core.

Paper -> code mapping (full table in notebook cell 0):
  Fig.4(b) "all K levels are predicted in one forward pass"  -> NCMLevelParallel.forward
  Fig.4(b) "block-diagonal attention restricts each input
            block to itself"                                -> LevelSchedule.dense_mask / flex_block_mask
  Sec.3.2  conditioning on the reconstruction z_hat(k-1)
            "resized to the l_k positions of level k"       -> NCMLevelParallel(conds=...)
  Sec.3.3  residual VQ + resampler R_k: l_k -> L            -> MultiscaleRVQVAE
  Table 6  GQA, QK-norm, RoPE, SwiGLU                       -> LevelAttention / Block

[DEV] per-level positions restart at 0 (VAR-style). The paper does not state the position
      scheme; this is a named choice, not a silent one.
[DEV] cross_level=True opens j<k blocks (block-causal instead of block-diagonal). Ablation
      knob only - the paper's default is block-diagonal. Off by default.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except Exception:                                     # pragma: no cover
    flex_attention = create_block_mask = None
    HAS_FLEX = False

# flex_attention is only fused (and only then faster than SDPA) under torch.compile;
# called eagerly it materialises the full score matrix and warns. Compile it once, lazily.
COMPILE_FLEX = os.environ.get("NCM_COMPILE_FLEX", "1") == "1"
_COMPILED_FLEX = None


def compiled_flex():
    """torch.compile(flex_attention, dynamic=False); permanently disabled on first failure."""
    global _COMPILED_FLEX, COMPILE_FLEX
    if _COMPILED_FLEX is None:
        try:
            _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
        except Exception:
            COMPILE_FLEX, _COMPILED_FLEX = False, False
    return _COMPILED_FLEX

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER = True
except Exception:
    HAS_LIGER = False


# --------------------------------------------------------------------------- schedule
@dataclass(frozen=True)
class LevelSchedule:
    """Level lengths l_1..l_K, optional conditional prefix (source/puzzle/null)."""
    level_lengths: Tuple[int, ...]
    prefix_len: int = 0

    def __post_init__(self):
        assert len(self.level_lengths) > 0 and all(l > 0 for l in self.level_lengths)

    @property
    def n_levels(self) -> int:
        return len(self.level_lengths)

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.level_lengths)

    def offsets(self) -> List[int]:
        out, c = [], self.prefix_len
        for l in self.level_lengths:
            out.append(c)
            c += l
        return out

    def level_ids(self, device="cpu") -> torch.Tensor:
        """-1 for prefix positions, k (0-based) for level-k positions."""
        ids = torch.full((self.total_len,), -1, dtype=torch.long)
        for k, (o, l) in enumerate(zip(self.offsets(), self.level_lengths)):
            ids[o:o + l] = k
        return ids.to(device)

    def pos_ids(self, device="cpu") -> torch.Tensor:
        p = torch.arange(self.prefix_len, dtype=torch.long, device=device)
        for l in self.level_lengths:
            p = torch.cat([p, torch.arange(l, dtype=torch.long, device=device)])
        return p

    def single_level(self, k: int) -> "LevelSchedule":
        return LevelSchedule((self.level_lengths[k],), self.prefix_len)

    # ------------------------------------------------------------------ masks
    def dense_mask(self, dtype=torch.float32, device="cpu", cross_level: bool = False) -> torch.Tensor:
        """Additive [1,1,T,T]: 0 = visible, -inf = blocked.

        Visible(q in level k, kv in level j) = (j is prefix) or (j == k).
        """
        ids = self.level_ids()
        q, kv = ids[:, None], ids[None, :]
        allowed = (kv == -1) | (kv == q)
        if cross_level:
            allowed = allowed | ((q >= 0) & (kv >= 0) & (kv < q))
        m = torch.zeros(self.total_len, self.total_len, dtype=dtype)
        m.masked_fill_(~allowed, float("-inf"))
        return m.view(1, 1, self.total_len, self.total_len).to(device)

    def flex_block_mask(self, B, H, device="cuda", cross_level: bool = False):
        assert HAS_FLEX, "flex_attention unavailable in this torch build"
        ids = self.level_ids(device)

        def mask_mod(b, h, q_idx, kv_idx):
            ql, kl = ids[q_idx], ids[kv_idx]
            allowed = (kl == -1) | (kl == ql)
            if cross_level:
                allowed = allowed | ((ql >= 0) & (kl >= 0) & (kl < ql))
            return allowed

        return create_block_mask(mask_mod, B=B, H=H, Q_LEN=self.total_len,
                                 KV_LEN=self.total_len, device=device)


# --------------------------------------------------------------------------- layers
def precompute_rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=positions.device) / half))
    f = positions.float()[:, None] * inv[None, :]
    return torch.cat([f.cos(), f.cos()], -1), torch.cat([f.sin(), f.sin()], -1)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class LevelAttention(nn.Module):
    """GQA attention: flex_attention+block_mask (fast) or SDPA+dense mask (fallback).

    Fast path keeps K/V at n_kv_head via enable_gqa - the reference implementation
    repeat_interleaves to n_head, materialising a full-size K/V copy every forward.
    """

    def __init__(self, n_embd, n_head, n_kv_head=None, dropout=0.0, qk_norm=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_kv_head = n_head, n_kv_head or n_head
        assert n_head % self.n_kv_head == 0
        self.hd, self.dropout = n_embd // n_head, dropout
        self.c_attn = nn.Linear(n_embd, n_embd + 2 * self.n_kv_head * self.hd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.q_norm = RMSNorm(self.hd) if qk_norm else None
        self.k_norm = RMSNorm(self.hd) if qk_norm else None

    def forward(self, x, block_mask=None, dense_mask=None, rope=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split([C, self.n_kv_head * self.hd, self.n_kv_head * self.hd], 2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            cos, sin = cos[:T].to(q.dtype), sin[:T].to(q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if block_mask is not None:
            fn = flex_attention
            if COMPILE_FLEX and q.is_cuda:
                cf = compiled_flex()
                if cf is not False:                  # False => compile unavailable
                    try:
                        y = cf(q, k, v, block_mask=block_mask,
                               enable_gqa=self.n_kv_head != self.n_head)
                        return self._head_out(y)
                    except Exception:
                        # e.g. head_dim < 16 (Inductor NYI), or a mask_mod it cannot trace
                        globals()["COMPILE_FLEX"] = False
            y = fn(q, k, v, block_mask=block_mask,
                   enable_gqa=self.n_kv_head != self.n_head)
        else:
            if self.n_kv_head != self.n_head:
                rep = self.n_head // self.n_kv_head
                k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            dm = None if dense_mask is None else dense_mask[..., :T, :T]
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=dm, dropout_p=self.dropout if self.training else 0.0,
                is_causal=False)
        return self._head_out(y)

    def _head_out(self, y):
        B, H, T, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, H * hd)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head=None, dropout=0.0, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(n_embd), RMSNorm(n_embd)
        self.attn = LevelAttention(n_embd, n_head, n_kv_head, dropout, qk_norm)
        h = int(2 * mlp_ratio * n_embd / 3 // 8 * 8)          # swiglu hidden width
        self.w_gate, self.w_up, self.w_down = (nn.Linear(n_embd, h), nn.Linear(n_embd, h),
                                               nn.Linear(h, n_embd))

    def forward(self, x, block_mask=None, dense_mask=None, rope=None):
        x = x + self.attn(self.ln1(x), block_mask=block_mask, dense_mask=dense_mask, rope=rope)
        n = self.ln2(x)
        return x + self.w_down(F.silu(self.w_gate(n)) * self.w_up(n))


# --------------------------------------------------------------------------- generator
class NCMLevelConfig:
    def __init__(self, schedule: LevelSchedule, codebook_size=64, n_embd=32, n_layer=2,
                 n_head=4, n_kv_head=2, d_latent=16, d_prompt=8, dropout=0.0,
                 qk_norm=True, max_level_len=None):
        self.schedule, self.codebook_size, self.n_embd = schedule, codebook_size, n_embd
        self.n_layer, self.n_head, self.n_kv_head = n_layer, n_head, n_kv_head
        self.d_latent, self.d_prompt, self.dropout = d_latent, d_prompt, dropout
        self.qk_norm = qk_norm
        self.max_level_len = max_level_len or max(max(schedule.level_lengths), schedule.prefix_len)


class NCMLevelParallel(nn.Module):
    """One transformer, all K levels predicted in a single forward (Fig.4b)."""

    def __init__(self, cfg: NCMLevelConfig):
        super().__init__()
        self.cfg, self.schedule = cfg, cfg.schedule
        K = self.schedule.n_levels
        self.level_emb = nn.Embedding(K, cfg.n_embd)
        self.tok_emb = nn.ModuleList([nn.Embedding(cfg.codebook_size, cfg.n_embd) for _ in range(K)])
        self.pos_emb = nn.Embedding(cfg.max_level_len + 1, cfg.n_embd)
        self.cond_proj = nn.Linear(cfg.d_latent, cfg.n_embd)     # z_hat(k-1) -> hidden
        self.prompt_proj = nn.Linear(cfg.d_prompt, cfg.n_embd)   # conditional prefix
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.dropout,
                                           qk_norm=cfg.qk_norm) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.heads = nn.ModuleList([nn.Linear(cfg.n_embd, cfg.codebook_size, bias=False)
                                    for _ in range(K)])

    # ------------------------------------------------------------------ internals
    def _rope(self, device):
        return precompute_rope_cos_sin(self.schedule.pos_ids(device),
                                       self.cfg.n_embd // self.cfg.n_head)

    def _tokens(self, k, codes_k, cond_k):
        l = self.schedule.level_lengths[k]
        assert codes_k.shape[-1] == l and cond_k.shape[1] == l, (k, codes_k.shape, cond_k.shape)
        return (self.tok_emb[k](codes_k) + self.cond_proj(cond_k) + self.level_emb.weight[k]
                + self.pos_emb(torch.arange(l, device=codes_k.device)))

    def _enc(self, x, kw, rope):
        for b in self.blocks:
            x = b(x, rope=rope, **kw)
        return self.ln_f(x)

    def _kw(self, B, device, sched, cross_level, use_flex):
        if use_flex is None:
            use_flex = HAS_FLEX and torch.device(device).type == "cuda"
        if use_flex:
            return {"block_mask": sched.flex_block_mask(B, self.cfg.n_head, device, cross_level)}
        return {"dense_mask": sched.dense_mask(torch.float32, device, cross_level)}

    # ------------------------------------------------------------------ fused path
    def forward(self, codes, conds, prefix=None, cross_level=False, use_flex=None):
        """All K levels in ONE forward pass -> list of per-level logits."""
        S, B = self.schedule, codes[0].size(0)
        parts = [] if not S.prefix_len else [self.prompt_proj(prefix)]
        for k in range(S.n_levels):
            parts.append(self._tokens(k, codes[k], conds[k]))
        x = torch.cat(parts, 1)
        h = self._enc(x, self._kw(B, x.device, S, cross_level, use_flex), self._rope(x.device))
        return [self.heads[k](h[:, o:o + l]) for k, (o, l) in enumerate(zip(S.offsets(), S.level_lengths))]

    # ------------------------------------------------------------------ oracle path
    def forward_sequential(self, codes, conds, prefix=None, cross_level=False, use_flex=None):
        """K separate forwards - the obvious implementation. Parity oracle, not a fallback."""
        S, B = self.schedule, codes[0].size(0)
        outs = []
        for k in range(S.n_levels):
            sub = S.single_level(k)
            parts = [] if not S.prefix_len else [self.prompt_proj(prefix)]
            parts.append(self._tokens(k, codes[k], conds[k]))
            x = torch.cat(parts, 1)
            rope = precompute_rope_cos_sin(sub.pos_ids(x.device), self.cfg.n_embd // self.cfg.n_head)
            h = self._enc(x, self._kw(B, x.device, sub, cross_level, use_flex), rope)
            outs.append(self.heads[k](h[:, S.prefix_len:S.prefix_len + S.level_lengths[k]]))
        return outs


# --------------------------------------------------------------------------- loss
def level_ce(logits, targets, chunk=None, label_smoothing=0.0):
    """Sum-of-levels CE. `chunk` splits the position axis; must match the single shot."""
    if HAS_LIGER and chunk is None:
        loss = logits[0].new_zeros(())
        for lg, tg in zip(logits, targets):
            loss = loss + LigerFusedLinearCrossEntropyLoss()(lg.reshape(-1, lg.size(-1)),
                                                            tg.reshape(-1, 1))
        return loss
    total, n = logits[0].new_zeros(()), 0
    for lg, tg in zip(logits, targets):
        flat_lg, flat_tg = lg.reshape(-1, lg.size(-1)), tg.reshape(-1)
        if chunk is None:
            total = total + F.cross_entropy(flat_lg, flat_tg, reduction="sum",
                                            label_smoothing=label_smoothing)
        else:
            part = flat_lg.new_zeros(())
            for i in range(0, flat_lg.size(0), chunk):
                part = part + F.cross_entropy(flat_lg[i:i + chunk], flat_tg[i:i + chunk],
                                              reduction="sum", label_smoothing=label_smoothing)
            total = total + part
        n += flat_tg.numel()
    return total / n


# --------------------------------------------------------------------------- tokenizer
class Resampler(nn.Module):
    """R_k: l_k -> L ('resized to the l_k positions of level k', Sec.3.3)."""

    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, e):                                 # [B,l_k,d] -> [B,L,d]
        return F.interpolate(e.transpose(1, 2), size=self.L, mode="nearest").transpose(1, 2)


class EMAQuantizer(nn.Module):
    def __init__(self, V, d, decay=0.99, eps=1e-5):
        super().__init__()
        self.register_buffer("codebook", torch.randn(V, d) / d ** 0.5)
        self.register_buffer("ema_count", torch.ones(V))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.decay, self.eps, self.V = decay, eps, V

    def forward(self, z, update=True):                    # z: [B,l,d]
        d2 = (z.pow(2).sum(-1, keepdim=True) - 2 * z @ self.codebook.T
              + self.codebook.pow(2).sum(-1))
        idx = d2.argmin(-1)
        if update and self.training:
            with torch.no_grad():                         # EMA update: no graph, no grad
                onehot = F.one_hot(idx.reshape(-1), self.V).to(z.dtype)
                flat = z.reshape(-1, z.size(-1)).detach()
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                self.codebook.copy_(self.ema_sum / (self.ema_count + self.eps).unsqueeze(1))
        zq = self.codebook[idx]
        return idx, zq, F.mse_loss(zq, z.detach())


class MultiscaleRVQVAE(nn.Module):
    """Toy multiscale residual VQ-VAE: text -> z [B,L,d] -> K levels of codes."""

    def __init__(self, vocab, schedule: LevelSchedule, d=16, codebook_size=64, decay=0.99):
        super().__init__()
        self.schedule, self.d, self.V = schedule, d, codebook_size
        self.L = schedule.level_lengths[-1]               # finest level == full length
        self.emb = nn.Embedding(vocab, d)
        self.enc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.quantizers = nn.ModuleList([EMAQuantizer(codebook_size, d, decay)
                                         for _ in schedule.level_lengths])
        self.resamplers = nn.ModuleList([Resampler(self.L) for _ in schedule.level_lengths])
        self.dec = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, vocab))

    def encode(self, x):
        return self.enc(self.emb(x))

    def quantize(self, z, level_cutoff=None):
        """codes[k] is [B,l_k] (-1 past the cutoff); zhat[k] is the cumulative L x d recon."""
        codes, zhat = [], []
        residual, zc = z.clone(), torch.zeros_like(z)
        for k, l in enumerate(self.schedule.level_lengths):
            if level_cutoff is not None and k >= level_cutoff:
                codes.append(torch.full((z.size(0), l), -1, dtype=torch.long, device=z.device))
                zhat.append(zc)
                continue
            zk = F.interpolate(residual.transpose(1, 2), size=l, mode="nearest").transpose(1, 2)
            idx, zq, _ = self.quantizers[k](zk)
            zc = zc + self.resamplers[k](zq)
            residual = z - zc
            codes.append(idx)
            zhat.append(zc)
        return codes, zhat

    def forward(self, x, level_cutoff=None):
        z = self.encode(x)
        codes, zhat = self.quantize(z, level_cutoff)
        return {"codes": codes, "zhat": zhat, "logits": [self.dec(zh) for zh in zhat], "z": z}

    def decode_codes(self, codes):
        zc = None
        for k, c in enumerate(codes):
            contrib = self.resamplers[k](self.quantizers[k].codebook[c])
            zc = contrib if zc is None else zc + contrib
        return self.dec(zc)


# --------------------------------------------------------------------------- sampling
@torch.no_grad()
def generate_codes(generator, tokenizer, prefix=None, B=1, device="cpu", cross_level=False):
    """Coarse-to-fine sampling: exactly K forwards, all l_k codes of a level in parallel."""
    g = generator.eval()
    S = g.schedule
    if prefix is not None and prefix.size(0) != B:      # a single prompt for a batch of B
        assert prefix.size(0) == 1, (prefix.shape, B)
        prefix = prefix.expand(B, -1, -1)
    codes, prev = [], None
    for k, l in enumerate(S.level_lengths):
        cond = (torch.zeros(B, l, tokenizer.d, device=device) if prev is None
                else F.interpolate(prev.transpose(1, 2), size=l, mode="nearest").transpose(1, 2))
        dummy = torch.zeros(B, l, dtype=torch.long, device=device)
        parts = [] if not S.prefix_len else [g.prompt_proj(prefix)]
        parts.append(g._tokens(k, dummy, cond))
        x = torch.cat(parts, 1)
        sub = S.single_level(k)
        kw = g._kw(B, x.device, sub, cross_level, None)
        rope = precompute_rope_cos_sin(sub.pos_ids(x.device), g.cfg.n_embd // g.cfg.n_head)
        h = g._enc(x, kw, rope)[:, S.prefix_len:]
        codes.append(g.heads[k](h).argmax(-1))
        zq = tokenizer.quantizers[k].codebook[codes[-1]]
        prev = tokenizer.resamplers[k](zq) if prev is None else prev + tokenizer.resamplers[k](zq)
    return codes
