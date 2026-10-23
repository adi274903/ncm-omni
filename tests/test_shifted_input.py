"""Can a position see its own target code?

The first Colab run trained to loss 7.7e-09 and then scored 0.122 at generation - below a
constant predictor (0.174). Cause: `pack` fed every position its OWN code, and that token
embedding is in the residual stream, so the head could copy the answer with a linear map. The
model learned identity, not a distribution, and at generation the input is NULL.

These assertions are the contract the fix has to hold, per honest arm:

  strict  level k's logits must not move when level k's codes change   (no self, no siblings)
          level k's logits must move when level k-1's codes change     (context path alive)
  ar      position j's logits must not move when code j changes        (raster shift, no self)
          position j's logits must move when code j-1 changes          (prefix path alive)
"""
import torch

from omni.model import OmniConfig, OmniNCM
from omni.schedule import OmniSchedule

torch.manual_seed(0)
sched = OmniSchedule(base_len=dict(text=8, image=8, audio=8, video=4), K=3, R=2)
vocab = {m: 257 for m in sched.modalities}
B = 2
present = {m: torch.ones(B, dtype=torch.bool) for m in sched.modalities}


def build(mode):
    cfg = OmniConfig(sched=sched, vocab=vocab, n_layer=2, n_embd=64, n_head=4, n_kv_head=1,
                     dropout=0.0, mask_mode=mode)
    return OmniNCM(cfg).eval()


def rnd():
    return {m: torch.randint(1, 257, (B, sched.flat_len(m))) for m in sched.modalities}


def at(model, codes, k):
    with torch.no_grad():
        out = model(dict(codes=codes, present=present), use_flex=False)
    return {m: out[(m, k)] for m in sched.modalities}


def delta(a, b):
    return max((a[m] - b[m]).abs().max().item() for m in sched.modalities)


def poke_level(codes, level):
    c = {m: v.clone() for m, v in codes.items()}
    for m in sched.modalities:
        off, l = sched.offsets(m)[level], sched.level_len(m, level)
        c[m][:, off:off + l] = torch.randint(1, 257, (B, l))
    return c


# ---------------------------------------------------------------- level-parallel
lineage = build("strict")
codes = rnd()
base = at(lineage, codes, 1)
d_self = delta(base, at(lineage, poke_level(codes, 1), 1))
d_ctx = delta(base, at(lineage, poke_level(codes, 0), 1))
print(f"[strict] perturb level 1 -> |dlogit| at level 1 = {d_self:.3e}  (want 0)")
print(f"[strict] perturb level 0 -> |dlogit| at level 1 = {d_ctx:.3e}  (want >0)")
assert d_self == 0.0, "level k still reads its own codes - the model can copy the answer"
assert d_ctx > 0.0, "level k cannot see level k-1 - the context path is dead"

g = lineage.generate(dict(codes=codes, present=present), level_cutoff=3, use_flex=False)
g2 = lineage.generate(dict(codes=codes, present=present), level_cutoff=3, use_flex=False)
assert all(g[m].shape == codes[m].shape for m in sched.modalities), "generate() shape drift"
assert all(torch.equal(g[m], g2[m]) for m in sched.modalities), "generate() not deterministic"
print(f"[strict] generate deterministic, shape {tuple(g['text'].shape)}")

# ---------------------------------------------------------------- autoregressive
def at_one(model, codes, level, j):
    """Logits for text position j only - a level-wide max would hide which token moved."""
    with torch.no_grad():
        out = model(dict(codes=codes, present=present), use_flex=False)
    off = sched.offsets("text")[level]
    return out[("text", level)][:, j - off]


arm = build("ar")
base_ar = at_one(arm, codes, 2, 8)
for which, tag, want_zero in ((8, "self j", True), (7, "prev j-1", False)):
    c = {m: v.clone() for m, v in codes.items()}
    c["text"][0, which] = (codes["text"][0, which] % 256) + 1      # different code, same token
    d = (base_ar - at_one(arm, c, 2, 8)).abs().max().item()
    print(f"[ar] poke one code at j={which} ({tag}) -> |dlogit| at position 8 = {d:.3e}")
    if want_zero:
        assert d == 0.0, f"position 8 reads its own code (j={which} moved it)"
    else:
        assert d > 0.0, "position 8 cannot see its predecessor - the prefix path is dead"

print("SHIFTED INPUT OK: no position can read its own target")
