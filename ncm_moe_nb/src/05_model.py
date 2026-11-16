# %% [markdown]
# ## 9. The model (training only)
#
# * **Embeddings**: one code table per modality (vocab 257) + learned absolute position per slot
#   (layouts are fixed-length and the image is 2D, so learned positions replace RoPE).
# * **Attention**: GQA 16/4, QK-RMSNorm, level-block-causal mask.
# * **Experts**: each modality *group* has 2 SwiGLU experts. A token only ever reaches its own
#   group (forced by modality tag, so no router collapse across modalities). Inside the group routing
#   is **expert-choice per level**: expert *e* takes the `ceil(cf * l)` tokens of the level it likes
#   best. Capacity is therefore exact, shapes are static, there is no load-balance loss, and a token
#   gets 0, 1 or 2 experts (1 on average at `cf=0.5`). Choosing among the tokens of **one level**
#   (never across levels) keeps routing from leaking information from finer levels.

# %%
from torch.utils.checkpoint import checkpoint as _ckpt


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w.to(x.dtype), self.eps)


_flex_fn = None
# Smaller tiles: the default 128x128 forward tile at head_dim 96 needs ~178 KB of shared memory, over the A100's 163 KB limit.
FLEX_OPTS = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32)
def _flex():
    global _flex_fn
    if _flex_fn is None:
        from torch.nn.attention.flex_attention import flex_attention
        _flex_fn = flex_attention if torch.compiler.is_compiling() else torch.compile(flex_attention)
    return _flex_fn


class Attn(nn.Module):
    def __init__(self, d, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, d // nh
        self.qkv = nn.Linear(d, (nh + 2 * nkv) * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, lay, use_flex):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = self.qn(q.view(B, T, self.nh, self.hd)).transpose(1, 2)
        k = self.kn(k.view(B, T, self.nkv, self.hd)).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.hd).transpose(1, 2)
        if ATTN == "levels":
            # block-causal over levels == for each level (and the prefix), plain unmasked attention over every key up to the end of that level
            r = self.nh // self.nkv
            segs = ([(0, lay.cond)] if lay.cond else []) + list(lay.lvl)
            y = torch.cat([F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, :b].repeat_interleave(r, 1), v[:, :, :b].repeat_interleave(r, 1))
                           for a, b in segs], 2)
        elif use_flex:
            y = _flex()(q, k, v, block_mask=lay.flex_mask, enable_gqa=True, kernel_options=FLEX_OPTS)
        else:
            r = self.nh // self.nkv
            y = F.scaled_dot_product_attention(q, k.repeat_interleave(r, 1), v.repeat_interleave(r, 1),
                                               attn_mask=lay.mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class GroupExperts(nn.Module):
    """E SwiGLU experts for one modality; expert-choice routing per segment (= per level)."""

    def __init__(self, d, h, E, cf):
        super().__init__()
        self.E, self.h, self.cf = E, h, cf
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.empty(E, d, 2 * h))      # gate | up, one GEMM
        self.w2 = nn.Parameter(torch.empty(E, h, d))
        self.record = False
        self.last_cov = None

    def forward(self, x, segs):
        B, T, d = x.shape
        E = self.E
        idxs, ws = [], []
        for a, b in segs:
            l = b - a
            c = max(1, math.ceil(self.cf * l))
            p = self.router(x[:, a:b]).float().softmax(-1)             # [B, l, E]
            w, i = p.transpose(1, 2).topk(c, dim=-1)                   # each expert's top-c tokens
            idxs.append(i + a); ws.append(w)
        idx, w = torch.cat(idxs, -1), torch.cat(ws, -1)               # [B, E, C]
        C = idx.shape[-1]
        if MOE_DISPATCH == "flat":
            # same maths, rows addressed in one flat table: one gather straight into the [E, B*C, d] layout and one index_add_ back
            tok = (idx + torch.arange(B, device=x.device).view(B, 1, 1) * T).permute(1, 0, 2).reshape(E * B * C)
            xg = x.reshape(B * T, d).index_select(0, tok).view(E, B * C, d)
            hh = swiglu(torch.bmm(xg, self.w1.to(xg.dtype)))          # [E, BC, h]
            y = torch.bmm(hh, self.w2.to(xg.dtype))                   # [E, BC, d]
            y = y * w.permute(1, 0, 2).reshape(E, B * C, 1).to(y.dtype)
            out = x.new_zeros(B * T, d).index_add_(0, tok, y.reshape(E * B * C, d).to(x.dtype)).view(B, T, d)
        else:
            xg = torch.gather(x.unsqueeze(1).expand(B, E, T, d), 2, idx.unsqueeze(-1).expand(B, E, C, d))
            xg = xg.permute(1, 0, 2, 3).reshape(E, B * C, d)
            hh = swiglu(torch.bmm(xg, self.w1.to(xg.dtype)))          # [E, BC, h]
            y = torch.bmm(hh, self.w2.to(xg.dtype)).view(E, B, C, d).permute(1, 0, 2, 3)
            y = y * w.unsqueeze(-1).to(y.dtype)
            out = x.new_zeros(B, T, d)
            out.scatter_add_(1, idx.reshape(B, E * C, 1).expand(B, E * C, d), y.reshape(B, E * C, d).to(out.dtype))
        if self.record:
            with torch.no_grad():
                cnt = torch.zeros(B, T, device=x.device).scatter_add_(1, idx.reshape(B, E * C), torch.ones(B, E * C, device=x.device))
                live = torch.zeros(T, dtype=torch.bool, device=x.device)
                for a, b in segs:
                    live[a:b] = True
                cl = cnt[:, live]
                self.last_cov = torch.stack([(cl == j).float().mean() for j in range(E + 1)])
        return out


class Block(nn.Module):
    def __init__(self, d, nh, nkv, h, E, cf):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attn(d, nh, nkv)
        self.experts = nn.ModuleDict({m: GroupExperts(d, h, E, cf) for m in ("text", "image", "audio")})

    def forward(self, x, lay, use_flex):
        x = x + self.attn(self.ln1(x), lay, use_flex)
        y = self.ln2(x)
        moe = None
        for g, segs in lay.segs.items():                 # only the groups this row type touches
            o = self.experts[g](y, segs)
            moe = o if moe is None else moe + o
        return x + moe


def up_codes(codes, m, spec, n_out=None):
    """Level k-1 codes -> level k grid (nearest). Image is 2D (2x2), text/audio are 1D. Same length (a residual level) = identity."""
    r = spec.R_of(m)                     # text pools x2, image 2x2, audio x4 - per modality
    if n_out is not None and codes.shape[1] == n_out:
        return codes
    if m == "image":
        B, n = codes.shape
        g, f = math.isqrt(n), math.isqrt(r)
        return codes.view(B, g, g).repeat_interleave(f, 1).repeat_interleave(f, 2).reshape(B, -1)
    return codes.repeat_interleave(r, 1)


class Model(nn.Module):
    def __init__(self, spec, d, layers, heads, kv_heads, ffn, cf, act_ckpt, E=2):
        super().__init__()
        self.spec, self.d, self.act_ckpt = spec, d, act_ckpt
        mods = ("text", "image", "audio")
        self.emb = nn.ModuleDict({m: nn.Embedding(spec.V, d) for m in mods})
        self.pos = nn.ParameterDict({m: nn.Parameter(torch.zeros(sum(spec.lens(m)), d)) for m in mods})
        self.pos_cond = nn.Parameter(torch.zeros(max(spec.cond, 1), d))
        self.blocks = nn.ModuleList([Block(d, heads, kv_heads, ffn, E, cf) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.heads = nn.ModuleDict({m: nn.Linear(d, spec.V) for m in mods})
        self.use_flex = False
        self.lay = None
        self.apply(self._init)
        resid = 0.02 / math.sqrt(2 * layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o.weight, std=resid)
            for g in blk.experts.values():
                nn.init.normal_(g.w1, std=0.02)
                nn.init.normal_(g.w2, std=resid)
                nn.init.normal_(g.router.weight, std=0.02)
        for p_ in list(self.pos.values()) + [self.pos_cond]:
            nn.init.normal_(p_, std=0.02)
        for hd in self.heads.values():                  # zero head: step-0 prediction is uniform, loss = ln(257) at any width
            nn.init.zeros_(hd.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def hidden(self, rtype, cond, codes):
        """cond [B, Lc] long | None, codes: K tensors [B, l_k] long -> [B, T, d].
        Level k's input is the upsampled codes of level k-1; level 0 reads the NULL code."""
        lay, m, spec = self.lay[rtype], ROW_MOD[rtype], self.spec
        B = codes[0].shape[0]
        parts = []
        if lay.cond:
            parts.append(self.emb["text"](cond) + self.pos_cond[:lay.cond])
        E, o = self.emb[m], 0
        for k, n in enumerate(lay.lens):
            ids = codes[0].new_zeros(B, n) if k == 0 else up_codes(codes[k - 1], m, spec, n)
            parts.append(E(ids) + self.pos[m][o:o + n])
            o += n
        x = torch.cat(parts, 1)
        if RESID == "bf16" and torch.is_autocast_enabled():          # fp32 residual = a cast + an fp32 norm around every matmul
            x = x.to(AMP_DTYPE)
        for blk in self.blocks:
            if self.act_ckpt and self.training and torch.is_grad_enabled():
                x = _ckpt(blk, x, lay, self.use_flex, use_reentrant=False)
            else:
                x = blk(x, lay, self.use_flex)
        return self.norm(x)

    def loss(self, rtype, cond, codes):
        lay, m = self.lay[rtype], ROW_MOD[rtype]
        h = self.hidden(rtype, cond, codes)[:, lay.cond:]
        B, Tt, d = h.shape
        tgt = torch.cat(codes, 1)
        head = self.heads[m]
        out = fused_ce(h.reshape(-1, d), head.weight, head.bias, tgt.reshape(-1))
        ce, lse = out[0].view(B, Tt), out[1].view(B, Tt)
        tok = ce + 1e-4 * lse * lse                                # light z-loss keeps logits bounded
        per_level = torch.stack([ce[:, a - lay.cond:b - lay.cond].mean() for a, b in lay.lvl]).detach()
        if LEVEL_W == "slots":
            loss = tok.mean()                                      # every position counts the same: a level's weight is its slot count
        else:                                                      # average inside each level first, then weight the levels (coarse levels are not drowned out)
            lvl = torch.stack([tok[:, a - lay.cond:b - lay.cond].mean() for a, b in lay.lvl])
            w = torch.tensor([float(b - a) ** (0.5 if LEVEL_W == "sqrt" else 0.0) for a, b in lay.lvl], device=lvl.device)
            loss = (lvl * w).sum() / w.sum()
        return loss, per_level

    def level_logits(self, rtype, h, k):
        lay, m = self.lay[rtype], ROW_MOD[rtype]
        a, b = lay.lvl[k]
        return self.heads[m](h[:, a:b]).float()


def build_model(meta=False):
    ctx = torch.device("meta") if meta else torch.device("cpu")
    with ctx:
        net = Model(SPEC, P["d"], P["layers"], P["heads"], P["kv_heads"], P["ffn"], MOE_CF, P["act_ckpt"] if ACT_CKPT < 0 else bool(ACT_CKPT))
    return net


def param_report(net):
    """total, trunk, per-expert, and active-per-token for each modality."""
    tot = sum(p.numel() for p in net.parameters())
    ex = {g: 0 for g in ("text", "image", "audio")}
    for blk in net.blocks:
        for g, mod in blk.experts.items():
            ex[g] += sum(p.numel() for n, p in mod.named_parameters() if n != "router.weight")
    experts_total = sum(ex.values())
    E = net.blocks[0].experts["text"].E
    per_expert = ex["text"] / E
    trunk = tot - experts_total
    act = {}
    for m in ex:
        own_emb = net.emb[m].weight.numel() + net.pos[m].numel() + sum(p.numel() for p in net.heads[m].parameters())
        other = sum(net.emb[o].weight.numel() + net.pos[o].numel() + sum(p.numel() for p in net.heads[o].parameters())
                    for o in ex if o != m)
        act[m] = trunk - other + ex[m] * (MOE_CF * E) / E
    return dict(total=tot, trunk=trunk, experts_total=experts_total, per_expert=per_expert,
                active=act, active_max=max(act.values()))
