# ncm_omni — streamed multimodal pipeline for level-parallel NCM

> **Start here: [STUDY.md](STUDY.md)** - training a 1.68B sparse MoE on one GPU (about 170 TFLOPS), what broke (audio overfit), the ablations, and what I'd change with 8 GPUs. Work in progress; no weights are published.


Level-parallel coarse-to-fine generation across **text + image + audio** (video in the data
path), built on the kernel-optimised core in `../ncm_fused/core.py`.

## What is here

| file | role |
|---|---|
| `omni/schedule.py` | level *k* = compression ratio `R**(K-1-k)` in **every** modality; level-major packing so one block = one altitude across modalities |
| `omni/codecs.py` | frozen codecs, all producing the exact schedule ladder: text bytes (finest decodes exactly), image patch pyramid, video frame×patch pyramid, audio log-mel rate pyramid over a fixed seeded codebook, span-hash fallback. `build_codec(..., tier=)` picks the tier and records it in `codec.info()["tier"]` |
| `omni/codecs_learned.py` | the learned tier: byte-entropy text segmentation, a post-hoc codebook tree over any flat codebook (Tree-VQ style), optional DINO+k-means image and DAC audio adapters. Falls back **visibly** — see `docs/codecs.md` |
| `omni/producer.py` | **streaming producer**: HF sources → codes → self-describing `.npz` shards, rolling buffer, back-pressure, deterministic resume, holdout per group, `EVAL_READY`/`DONE`/`STOP` |
| `omni/data.py` | `ShardStream` (consumes and deletes shards), collate to the packed batch contract, synthetic batches |
| `omni/model.py` | `OmniNCM`: one transformer, block-diagonal mask (parallel arm) or causal raster (AR arm), level-conditioned by `gt_embed`; `selfcheck()` = 7 invariants |
| `omni/train.py` | loop, val per (modality, level), checkpoints, `--smoke` prints s/step + projected $ |
| `omni/eval.py` | quality-vs-NFE table for both arms, teacher-free greedy, held-out |
| `run_omni.py` | `check / cfg / produce / train / eval / all` |

## Run order (Colab)

```bash
python run_omni.py check                 # invariants; needs no data
python run_omni.py cfg                   # writes run/cfg/*.json for both arms
python -m omni.producer run/cfg/produce.json &   # or use the notebook's cell 3
python run_omni.py train --smoke 10      # s/step + projected cost
python run_omni.py train                 # level-parallel arm
python run_omni.py train --arm ar        # matched autoregressive arm
python run_omni.py eval                  # the table
```

`core.py` must be importable: either copy it next to `run_omni.py`, or set `NCM_CORE_DIR`.
The model imports `Block`, `RMSNorm`, `precompute_rope_cos_sin`, `HAS_FLEX` from it.

## Codecs: which tier produced the shards

`produce.json` has `codec_tier`: `auto` (default), `learned` or `standin`. `auto` uses a learned
codec when this machine can build one — text **always** (byte-entropy segmentation), image if a
k-means codebook exists, audio if `descript-audio-codec` is installed — and otherwise the
dependency-free stand-in. Whichever wins, plus the reason it fell back, is printed by
`run_omni.py check` and written into every shard's `meta.codecs`. `standin` reproduces an existing
run byte-for-byte.

This means **shards are only comparable within one tier**: `auto` on a bare box gives text the
entropy codec and image/audio the stand-ins, which is not the same data a Colab run with
`transformers`/`dac` produces. Regenerate rather than mix. Which published codec each choice
corresponds to, why, and what is still missing: **`docs/codecs.md`**.

## Design decisions that are load-bearing

1. **Levels are bound to compression, not to length.** `l[m][k] = base_len[m] / R**(K-1-k)`.
   Otherwise "level 3 of text" and "level 3 of audio" have no shared referent and the
   cross-modal attention at a level is meaningless.
2. **Level-major packing.** One block per level containing every modality's slot; the mask is
   block-diagonal, so level *k* sees all of level *k* across modalities and nothing finer.
   `cross_level=True` is the ablation that tests whether that restriction is what buys the speed.
3. **Conditioning is `gt_embed`.** NCM conditions level *k* on the codec's reconstruction
   `z_hat(k-1)`; our codecs are discrete-only, so the model conditions on the mean embedding of
   the level-(k-1) codes actually fed in. Deviations to state in the paper.
4. **Absent modalities are NULL, not dropped.** Constant batch width, loss masked. They still
   occupy attention slots — cheap, and it keeps the pack shape static.
5. **Audio is the weak link, by construction.** Its hierarchy is temporal *rate*; pooling a
   spectrogram 32x destroys formants. If audio's fine level sags while text/image hold, that is
   the finding, not a bug. The published fix is a dual-stream semantic/acoustic codec at a fixed
   low frame rate plus variable-rate budgeted events (DualCodec 12.5 Hz, RubatoCodec) —
   `docs/codecs.md` §4, not a tuning knob.
6. **Codes are 1-based; 0 is NULL.** Every codec emits `1..codebook_size`, so `vocab` is
   `codebook_size + 1` and a real code can never be confused with padding. Audio needs this most:
   its codes are `argmax` indices, which start at 0.
7. **Video source ids are the least stable part of the package.** `_first([...])` tries several
   and logs the failures, so a renamed dataset costs the video group and nothing else. Video
   needs `av` (or pre-decoded frames) in the dataset; if the decode fails the sample is dropped.

## Running on Colab

The package is cloned from the private repo, never uploaded by hand:

```
!git clone --depth 1 https://x-access-token:$GITHUB_TOKEN@github.com/adi274903/ncm-omni.git /content/ncm_omni
```

`NCM_Omni_Stream.ipynb` cell 1 does exactly that, reading `GITHUB_TOKEN` from a Colab secret
(key icon, left sidebar), then from the environment, then from `gh auth token`. It strips the
token from the remote afterwards so nothing on disk keeps it. Re-running cell 1 becomes a
`git pull --ff-only`, so a code change reaches Colab without losing the shard buffer.

## Three arms, and one that cheats

Every position used to be fed **its own target code**, and that embedding sits in the residual
stream, so the head could copy the answer with a linear map. The model learned identity: train
loss hit `7.7e-09` in 2k steps, and at generation - where the input is NULL or a guess - it
scored `0.122`, **below the constant-predictor baseline of `0.174`**. That run is void.

`pack()` now shifts the input so no position can read its own target:

| arm | input at a position | attention | status |
|---|---|---|---|
| `strict` | codes of level k-1, upsampled to l_k (level 0: NULL) | levels < k only | the honest parallel arm |
| `ar` | code j-1 in raster order | causal | the honest AR arm |
| `block` | its own level's codes | own level only | **legacy, leaks, do not quote** |

`tests/test_shifted_input.py` asserts the contract: perturbing level k leaves level-k logits
bit-identical, perturbing level k-1 moves them; for AR, perturbing code j leaves position j
untouched and perturbing code j-1 moves it.

`eval` also prints the constant-predictor baseline, computed from the training shards (absent
modalities excluded). A score under it is not a result.

## Honest status

Authored, **not executed** (the machine that wrote it was off-limits for running). Every module
ships the assertions that would have caught its own bugs on the target GPU:

- `omni/model.py::selfcheck()` — 7 invariants including "level k cannot see k+1" and "AR mode
  does not leak the future".
- `omni/producer.py --selfcheck` — encodes two samples, writes a shard, checks shapes against
  the schedule.
- `run_omni.py check` — runs all of the above plus the real model build and collate.

Run `check` first; it fails loudly and cheaply. Until `check` passes and the model card matches
`core.py`, treat `omni/` as unvalidated.
