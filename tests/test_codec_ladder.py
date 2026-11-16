"""Does every codec, at every tier, hold the ladder contract?

The producer's selfcheck proves the contract for whatever `build_codec` returned on THAT machine.
This file proves it for the tiers it can build here, plus the two properties the tree hierarchy
adds and that no assertion in the pipeline would catch:

  contract   len(level_k) == schedule.level_len(m, k); every code in 1..V; 0 is only NULL
  tree       a coarse level is an ANCESTOR of the finest one - not an independent re-encode.
             Concretely: coarsen(fine_code) == coarse_code, and repeated coarsening is stable.
  honesty    a tier that cannot be built does not silently become something else: the codec says
             `tier=standin` and carries the reason in `.note` / `info()["note"]`.

Run: python tests/test_codec_ladder.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from omni.codecs import build_codec                                                # noqa: E402
from omni.codecs_learned import (TextEntropyCodec, TreeCodebookCodec,             # noqa: E402
                                 TreeHierarchy)
from omni.schedule import OmniSchedule                                            # noqa: E402

K, R, V = 6, 2, 256
BASE = dict(text=64, image=64, audio=96, video=32)
sched = OmniSchedule(base_len=BASE, K=K, R=R)


def _img(seed, side=32):
    from PIL import Image
    g = np.random.default_rng(seed)
    return Image.fromarray((g.random((side, side, 3)) * 255).astype("uint8"))


def _samples():
    return {
        "text": "a blood pressure of 140 over 90 is stage two hypertension",
        "image": _img(0),
        "audio": (np.sin(np.linspace(0, 400, 16000)).astype("float32"), 16000),
        "video": [_img(i, 16) for i in range(8)],
    }


def _check_contract(codec, m, obj, tag):
    levels = codec.encode(obj)
    assert len(levels) == K, f"{tag}/{m}: {len(levels)} levels, schedule wants {K}"
    for k, arr in enumerate(levels):
        want = sched.level_len(m, k)
        assert len(arr) == want, f"{tag}/{m} level {k}: len {len(arr)} != {want}"
        arr = np.asarray(arr)
        # The producer clips into 1..V because 0 is the NULL slot and text pads its finest rung with
        # zeros when the string is shorter than base_len. Anything outside 0..V is a codec bug.
        assert arr.min() >= 0, f"{tag}/{m} level {k}: negative code {int(arr.min())}"
        assert arr.max() <= codec.V, f"{tag}/{m} level {k}: code {int(arr.max())} > V {codec.V}"
        assert np.clip(arr, 1, codec.V).max() >= 1, f"{tag}/{m} level {k}: no live code"
    again = codec.encode(obj)
    for k in range(K):
        assert np.array_equal(np.asarray(levels[k]), np.asarray(again[k])), \
            f"{tag}/{m} level {k}: encode is not deterministic"
    return levels


# ------------------------------------------------------------------ 1. contract, per tier
for tier in ("standin", "auto"):
    samples = _samples()
    for m in ("text", "image", "audio", "video"):
        codec = build_codec(m, K, R, V, BASE[m], prefer_real=True, tier=tier)
        _check_contract(codec, m, samples[m], tier)
        info = codec.info()
        assert info["tier"], f"{tier}/{m}: info() has no tier"
        assert info["tier"] in ("standin", "bytes", "bytes+entropy", "learned"), info["tier"]
        print(f"  contract {tier:8s} {m:5s} {codec.name:20s} tier={info['tier']}")

# `tier=learned` must either build or say why, never quietly hand back a stand-in
for m in ("image", "audio"):
    try:
        c = build_codec(m, K, R, V, BASE[m], tier="learned")
        assert c.tier != "standin", f"{m}: tier=learned returned a stand-in {c.name}"
    except RuntimeError as e:
        assert "unavailable" in str(e), f"{m}: unhelpful failure {e}"
        print(f"  learned  {m:5s} unavailable (expected here): {str(e)[:80]}")

# ------------------------------------------------------------------ 2. tree: a level is an ancestor
rng = np.random.default_rng(0)
leaves = rng.standard_normal((64, 8)).astype("float32")
tree = TreeHierarchy(leaves)
depths, note = tree.plan(5, 256)
assert len(depths) == 5 and depths == sorted(depths, reverse=True), f"depths not coarse->fine: {depths}"
answers = [tree.n_nodes(d) for d in depths]
assert answers[0] <= answers[-1], f"nodes must shrink toward the coarse end: {answers}"
assert max(answers) <= 256, f"a level exceeded the codebook cap: {answers}"
assert note == "", f"unexpected plan note for a clean codebook: {note}"

fine_codes = tree.level_codes(depths[-1])
for k in range(5):
    coarse = tree.level_codes(depths[k])
    mapped = tree.coarsen(depths[-1], depths[k])
    assert np.array_equal(mapped[fine_codes - 1], coarse), \
        f"level {k}: coarse code is not the ancestor of the finest code"

# sibling mapping must be consistent: coarse(A) == coarse(B) iff they share an ancestor
top = tree.coarsen(depths[-1], depths[0])
assert len(np.unique(top)) == tree.n_nodes(depths[0]), "two coarse cells collapsed or split"
# climbing again is idempotent (a cell only widens)
again = tree.coarsen(depths[1], depths[0])
assert np.array_equal(np.unique(again), np.unique(top)), "coarsening chain disagrees"
print(f"  tree     depths={depths} nodes={answers}")


class _Toy(TreeCodebookCodec):
    """Tree codec over raw [n, D] features, so the hierarchy can be tested without a backbone."""

    def _feats(self, obj):
        return np.asarray(obj, dtype="float32")


toy = _Toy(K=5, R=2, V=V, base_len=16, leaves=leaves)
feats = rng.standard_normal((16, 8)).astype("float32")
lv = toy.encode(feats)
assert [len(a) for a in lv] == [toy.level_len(k) for k in range(5)], [len(a) for a in lv]
assert all(np.asarray(a).min() >= 1 for a in lv), "toy tree codec emitted a NULL code"
dec = toy.decode(lv[-1])
assert dec.shape[0] == len(lv[-1]) and dec.shape[1] == 8, f"decode shape {dec.shape}"
print(f"  tree     toy codec nodes={[toy.tree.n_nodes(d) for d in toy.depths]}")

# ------------------------------------------------------------------ 3. text: lossless bottom, word cuts
te = TextEntropyCodec(K=K, R=R, codebook_size=V, base_len=64)
txt = "a blood pressure of 140 over 90 is stage two hypertension"
enc = te.encode(txt)
assert te.decode(enc) == txt, f"finest rung is not lossless: {te.decode(enc)!r}"
raw = txt.encode()
edges = te._segments(raw, 4)
words = [raw[a:b].decode() for a, b in zip(edges, edges[1:])]
assert len(edges) - 1 == 4, edges
# every cut sits on a word boundary: a coarse span may end with the space, never mid-word
assert all(raw[b - 1:b] in (b" ", b"\t", b"\n", b"\r") for b in edges[1:-1]), \
    f"a coarse span cuts a word: {edges} {words}"
assert te._segments(raw, 4) == edges, "segmentation is not deterministic"
empty = te.encode("")
assert [len(a) for a in empty] == [te.level_len(k) for k in range(K)], [len(a) for a in empty]
assert all(np.asarray(a).max() <= V for a in empty), "empty text overshot the codebook"
print(f"  text     4 coarse spans = {words}")

# ------------------------------------------------------------------ 4. fallback is visible
auto_img = build_codec("image", K, R, V, BASE["image"], tier="auto")
if auto_img.tier == "standin":
    assert auto_img.note, "fell back to a stand-in without saying why"
    print(f"  honesty  image fell back: {auto_img.note[:80]}")

print("PASS test_codec_ladder")
