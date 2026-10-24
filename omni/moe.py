"""Modality specialists: one FFN group per modality, routed by modality tag, not by a router.

Why forced routing instead of a learned gate: the training mix is 70% text / 20% vision / 8%
audio / 2% video. A learned router collapses onto text within a few thousand steps at that
imbalance, and the collapse is silent - the loss looks fine because text is most of the loss.
Tagging the expert group by modality removes the failure mode entirely and turns load balance
into a data-mix decision instead of a training bug.

What this buys: the trunk (attention, embeddings, norms) is shared, so levels still talk across
modalities at the same altitude - that is the omni thesis - while each modality's feed-forward
capacity is its own. Text and audio stop fighting over one FFN, which is what the 0.92-vs-0.26
finest-level gap in the first eval was measuring.

Video has no expert of its own: a video sample is a stack of frames encoded as images, so it has
no separate code distribution to specialise on. It routes to the image expert by default.

Active parameters per token = trunk + one expert. Total parameters = trunk + n_experts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# video -> image: frames are images, not a modality (see the module docstring).
DEFAULT_EXPERT_MAP = {"text": 0, "image": 1, "audio": 2, "video": 1}


def resolve_expert_map(modalities, mapping=None, n_experts=None):
    """-> (per-modality expert index list in `modalities` order, n_experts)."""
    mp = dict(DEFAULT_EXPERT_MAP if mapping is None else mapping)
    idx = [int(mp.get(m, 0)) for m in modalities]
    n = int(n_experts or (max(idx) + 1))
    if n < 1:
        raise ValueError(f"n_experts must be >= 1, got {n}")
    return [min(max(e, 0), n - 1) for e in idx], n


class ModalityExperts(nn.Module):
    """n_experts independent SwiGLU FFNs; every token goes to exactly one of them."""

    def __init__(self, n_embd, n_experts, mlp_ratio=4.0):
        super().__init__()
        h = int(2 * mlp_ratio * n_embd / 3 // 8 * 8)          # same width as core.Block
        self.n_experts = n_experts
        self.w_gate = nn.ModuleList([nn.Linear(n_embd, h) for _ in range(n_experts)])
        self.w_up = nn.ModuleList([nn.Linear(n_embd, h) for _ in range(n_experts)])
        self.w_down = nn.ModuleList([nn.Linear(h, n_embd) for _ in range(n_experts)])

    def forward(self, x, mod_id, plan=None):
        """x = [B, T, d], mod_id = [T] expert index per packed position -> [B, T, d].

        `plan` is the per-expert position index, precomputed once with the routing tensor. Routing
        is a pure function of the schedule, so it never changes between steps: computing
        `(mod_id == e).nonzero()` here instead would be 3 kernels per block per step, and the
        `if not bool(sel.any()): continue` this replaces was a device sync plus a dynamo graph
        break in the middle of the forward - same bug the training loss had.
        """
        B, T, d = x.shape
        if mod_id is None:
            raise ValueError("ModalityExperts needs mod_id; forced routing has no fallback")
        if mod_id.dim() != 1 or mod_id.numel() != T:
            raise ValueError(f"mod_id must be [T] with T={T}, got {tuple(mod_id.shape)}")
        if plan is None:
            plan = tuple((mod_id == e).nonzero(as_tuple=False).squeeze(1)
                         for e in range(self.n_experts))
        out = x.new_zeros(B, T, d)
        for e, idx in enumerate(plan):
            if idx.numel() == 0:
                continue                                   # expert got no tokens: host-side shape
            xe = x.index_select(1, idx)
            ye = self.w_down[e](F.silu(self.w_gate[e](xe)) * self.w_up[e](xe))
            out.index_copy_(1, idx, ye.to(out.dtype))      # autocast emits bf16, out is fp32
        return out

    @torch.no_grad()
    def usage(self, mod_id):
        """Token count per expert - logged so a dead expert is visible, not inferred."""
        return torch.bincount(mod_id, minlength=self.n_experts).tolist()


class MoEBlock(nn.Module):
    """A core.Block with its dense FFN swapped for per-modality experts.

    Takes ownership of the block's attention and norms and drops the original FFN parameters
    instead of leaving them dead, so the parameter count means what it says. Submodule names are
    kept (ln1, ln2, attn), so state_dict keys stay blocks.<i>.attn... as before.
    """

    def __init__(self, base, n_embd, n_experts, mlp_ratio=4.0):
        super().__init__()
        self.ln1, self.ln2, self.attn = base.ln1, base.ln2, base.attn
        for name in ("w_gate", "w_up", "w_down"):
            if hasattr(base, name):
                delattr(base, name)                        # frees the dense FFN, not just ignores it
        self.experts = ModalityExperts(n_embd, n_experts, mlp_ratio)

    def forward(self, x, block_mask=None, dense_mask=None, rope=None, mod_id=None, plan=None):
        x = x + self.attn(self.ln1(x), block_mask=block_mask, dense_mask=dense_mask, rope=rope)
        return x + self.experts(self.ln2(x), mod_id, plan)


def active_params(model):
    """Trunk + one expert, i.e. what a forward pass actually touches per token."""
    total = sum(p.numel() for p in model.parameters())
    experts = 0
    for mod in model.modules():
        if isinstance(mod, ModalityExperts):
            per = sum(p.numel() for p in mod.parameters()) / mod.n_experts
            experts += sum(p.numel() for p in mod.parameters()) - per   # keep just one expert
    return total - experts
