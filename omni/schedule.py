"""Modality-aware level schedule: level k = the same semantic altitude in every modality.

The paper's text schedule makes level k have l_k positions. For omni we cannot give two
modalities the same *length*, because 30 s of audio and 64 characters of text are not
comparable. What we can hold fixed is the COMPRESSION RATIO: level k in every modality is
the input coarsened by R**(K-1-k). That is the binding that makes cross-modal attention at a
level mean something. Lengths are then derived per modality:

    l[m][k] = max(1, round(base_len[m] / R**(K-1-k)))

Packing order is level-major: [level 0: text|image|audio|video][level 1: ...]...[level K-1].
So one block = one altitude across all modalities, and the block-diagonal mask lets level k
see the whole level-k block (every modality) but nothing finer. That single line of mask code
is the entire omni thesis; `cross_level=True` turns it into the ablation.
"""
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import torch

# Video is deliberately absent: a frame is an image, so video samples are encoded as
# image samples and there is no separate code distribution to model or route.
MODALITIES = ("text", "image", "audio")


@dataclass
class OmniSchedule:
    base_len: Dict[str, int]                 # finest-level length per modality
    K: int = 6
    R: int = 2
    null_code: int = 0                       # reserved in every modality's codebook
    modalities: Tuple[str, ...] = MODALITIES
    prefix_len: int = 0

    def __post_init__(self):
        assert self.R >= 2 and self.K >= 1
        for m in self.modalities:
            assert m in self.base_len, m
        # level 0 is the COARSEST (short) and level K-1 the finest (== base_len), so lengths must
        # be non-decreasing in k. Getting this direction wrong is exactly what `check` caught.
        for m in self.modalities:
            lens = self.level_lens(m)
            assert all(a <= b for a, b in zip(lens, lens[1:])), (m, lens)

    # ---------------------------------------------------------------- geometry
    def level_len(self, m: str, k: int) -> int:
        return max(1, round(self.base_len[m] / self.R ** (self.K - 1 - k)))

    def level_lens(self, m: str) -> Tuple[int, ...]:
        return tuple(self.level_len(m, k) for k in range(self.K))

    def block_width(self, k: int) -> int:
        return sum(self.level_len(m, k) for m in self.modalities)

    def block_offsets(self) -> Tuple[int, ...]:
        out, c = [], self.prefix_len
        for k in range(self.K):
            out.append(c)
            c += self.block_width(k)
        return tuple(out)

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.block_width(k) for k in range(self.K))

    def offsets(self, m: str) -> Tuple[int, ...]:
        """Within-modality flat offset of each level, used by the shard layout."""
        out, c = [], 0
        for l in self.level_lens(m):
            out.append(c)
            c += l
        return tuple(out)

    def flat_len(self, m: str) -> int:
        return sum(self.level_lens(m))

    def positions(self):
        """Per packed position: (level k, modality index, position within that modality-level)."""
        lev, mod, pos = [], [], []
        for k in range(self.K):
            for mi, m in enumerate(self.modalities):
                l = self.level_len(m, k)
                lev += [k] * l
                mod += [mi] * l
                pos += list(range(l))
        if self.prefix_len:
            lev = [-1] * self.prefix_len + lev
            mod = [len(self.modalities)] * self.prefix_len + mod
            pos = list(range(self.prefix_len)) + pos
        return (torch.tensor(lev), torch.tensor(mod), torch.tensor(pos))

    # ---------------------------------------------------------------- masks
    def dense_mask(self, dtype=torch.float32, device="cpu", cross_level=False) -> torch.Tensor:
        lev, _, _ = self.positions()
        q, kv = lev[:, None], lev[None, :]
        allowed = (kv == -1) | (kv == q)
        if cross_level:
            allowed = allowed | ((q >= 0) & (kv >= 0) & (kv < q))
        m = torch.zeros(self.total_len, self.total_len, dtype=dtype)
        m.masked_fill_(~allowed, float("-inf"))
        return m.view(1, 1, self.total_len, self.total_len).to(device)

    def flex_block_mask(self, B, H, device="cuda", cross_level=False):
        from torch.nn.attention.flex_attention import create_block_mask
        lev, _, _ = self.positions()
        lev = lev.to(device)

        def mask_mod(b, h, q_idx, kv_idx):
            ql, kl = lev[q_idx], lev[kv_idx]
            allowed = (kl == -1) | (kl == ql)
            if cross_level:
                allowed = allowed | ((ql >= 0) & (kl >= 0) & (kl < ql))
            return allowed

        return create_block_mask(mask_mod, B=B, H=H, Q_LEN=self.total_len,
                                 KV_LEN=self.total_len, device=device)

    def describe(self) -> str:
        rows = []
        for m in self.modalities:
            rows.append(f"{m:6s} base={self.base_len[m]:4d} levels={self.level_lens(m)}")
        tot = {m: self.flat_len(m) for m in self.modalities}
        return ("K=%d R=%d  total packed positions=%d\n" % (self.K, self.R, self.total_len)
                + "\n".join(rows) + f"\nper-sample token counts {tot}")
