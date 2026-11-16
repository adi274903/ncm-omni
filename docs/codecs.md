# Which codec to train NCM on: image, audio, text

Scope: what to feed `omni.codecs` so the K-level ladder carries *concept* at the coarse rungs and
*detail* at the fine ones, per modality. Every number below is quoted from the cited paper; nothing
here was measured in this repo. Where the repo still uses a stand-in, the table says so.

The invariant this document serves is the one in `README.md`: level *k* is the input compressed by
`R**(K-1-k)`, the same altitude in every modality, and level *k* conditions level *k+1* through the
codec's reconstruction (`gt_embed`). A codec is good here iff (a) its finest rung is decodable,
(b) its coarse rungs are *coarsenings of the same latent*, not independent re-encodes, and (c) the
codes are predictable by a small transformer — "reconstructs well" is neither necessary nor
sufficient for (c).

---

## 1. Verdict table

| modality | default now (`tier=auto` on a bare box) | recommended target | why in one line | cite |
|---|---|---|---|---|
| text | `text_entropy` (bytes bottom, word-aligned novel spans above) | bytes bottom + **learned concept boundaries** | byte LMs beat subword LMs as scale grows and *learn* the segmentation you otherwise hand-code | 2610.05978, 2512.24617 |
| image | `image_pyramid` (stand-in) | discrete **feature-loss VQ** + **codebook tree** or **quadtree** | tokens should be easy for a generator to model, and a tree makes every prefix a valid latent | 2610.22060, 2609.03641, 2610.10497 |
| audio | `audio_mel_pyramid` (stand-in) | **dual-stream semantic→acoustic** at a fixed low frame rate, **variable-rate budgeted** | audio's hierarchy is temporal rate; rate-pooling a spectrogram destroys formants | 2609.08703, 2606.06743, 2610.29434 |
| video | `video_pyramid` (stand-in, data-path only) | image codec per frame; joint audio-video only if video ever becomes a target | a frame is an image | 2606.30811 |

Fallbacks that are still worth having as baselines: fidelity-first DAC / X-Codec-2.0 for audio
(2601.20185), 16-channel continuous VAE (SD3/Flux family) or a representation autoencoder (RAE /
RAEv2) for image if the diffusion arm is ever revived, and BPE for text **only** as an ablation.

---

## 2. Text

**Keep UTF-8 bytes at the finest rung.** `TextByteCodec` already does this and it is not a
compromise: 2610.05978 ("Byte Language Models: Scaling, Emergent Abstractions, and Information
Allocation") trains byte Transformers with token superposition + hash embeddings and finds they
**consistently outperform subword Transformers as model size scales**, that they acquire a set of
segmentation-like positions used to collect local context, and that restricting up to 25% of
intermediate layers to those positions preserves downstream performance. Bonus result relevant to
your eval: uncertainty concentrates near those learned boundaries, which yields 3.4× more accepted
tokens in speculative decoding than subword models.

**Coarse rungs should be concept boundaries, not fixed-width hashes.** `TextByteCodec` currently
hashes `R^k` byte spans, which cuts words and sentences at arbitrary offsets and forces the coarse
levels to be memorised noise. Two 2026 directions:

- 2512.24617 **Dynamic Large Concept Models (DLCM)**: learn variable-length concepts end-to-end
  from latent representations, move computation from tokens to a compressed concept space, and — the
  part you want for the paper — introduce a **compression-aware scaling law** that separates
  token-level capacity, concept-level capacity and compression ratio `R`, plus a decoupled μP
  parametrisation for zero-shot hyperparameter transfer across widths and `R`. At `R=4` it moves
  ~⅓ of inference compute into a higher-capacity reasoning backbone for +2.69% average over 12
  zero-shot benchmarks at matched FLOPs. This is the closest published relative of level-parallel
  NCM and the right thing to cite next to your `R` knob.
- 2610.26695: a hierarchical token-merging autoencoder with a variational information bottleneck —
  the hierarchy emerges from **compression alone** (93% parse validity, 93% exact reconstruction on
  a synthetic ambiguous grammar), which is the strongest available evidence that "level k = input
  compressed by R^(K-1-k)" is a sound *inductive bias* and not just a bookkeeping device.
- 2610.26225: treat the BPE merge budget as **model selection** (information criteria from corpus
  statistics). Use it if you need a subword baseline that is not strawmanned.

**What is implemented** (`TextEntropyCodec`): lossless bytes at `K-1`; at level *k* the text is cut
into `l_k` spans at *word starts*, spending the cuts first where the upcoming 4-gram is least
repeated. Deterministic, corpus-free, no training, and the finest rung still decodes exactly. It is
a proxy for a boundary predictor, not one — see §6 for the honest limit.

**Caution to state in the paper**: 2610.10304 (SemanticFold) compresses prefix hidden states at
learned boundaries across five model scales and finds NLL, linear-probe decodability and reasoning
accuracy move **non-monotonically and with different thresholds** under the same compression, so
there is *no single scalar certificate* that a compressed level "preserved" the information. Report
per-level NLL **and** a probe **and** a behavioural metric; do not claim level-k preservation from a
reconstruction number.

---

## 3. Image

Three separate decisions: what the leaf codebook is, how levels are derived from it, and what the
tokenizer is optimised for.

**(a) Optimise for the generator, not for pixels.** 2610.22060 **FeaTok** trains *discrete* tokens
to reconstruct pretrained features (SigLIP/SigLIP2/DINOv2/DINOv3) instead of pixels: −29% gFDr6 at
256² vs the strongest discrete baselines, and it beats the strongest *continuous* baseline (RAEv2)
with **38× less sampling compute**; human-preference reward scores improve without reward training.
This is the mechanical reason your level-0 should carry a concept rather than a blur.

**(b) How to get coarse-to-fine.** Four constructions, in increasing order of how much they respect
the "coarsening of one latent" requirement:

| construction | what level k is | use when | source |
|---|---|---|---|
| (a) pool the finest | mean-pool features to `l_k`, then quantise | stand-in default; always valid, loses detail by averaging | this repo |
| (b) RVQ stage as level | level k = k-th residual stage | you want exact rate control per rung; stages are not spatially nested | EnCodec/DAC lineage |
| (c) **post-hoc codebook tree** | level k = ancestor node of the finest code, K-1-k merges up | **the default recommendation**: any flat codebook, no re-encode, every prefix is a valid latent | 2609.03641 |
| (d) quadtree / content-adaptive | level k = tree node; capacity follows image content | variable `l_k` per image is acceptable | 2610.10497, 2610.29434 (audio) |

2609.03641 **Tree-VQ** is the single most reusable idea for your ladder: a pretrained *flat* VQ
tokenizer is converted, post hoc, into a balanced binary hierarchy with explicit internal-node
representations and depth-major branch decisions, so **any payload prefix uniquely specifies a valid
latent state** while every learned leaf vector and encoder assignment is preserved — 52.1% DISTS
BD-rate saving over ProGIC with thousands of usable truncation points from one bitstream. Your
`level_len(m,k)` can stay exactly as specified: only the *code* changes with depth, never the
segmentation of the payload.

2610.10497 **QuadTok** is the alternative when you want length adaptivity: a quadtree tokenizer
allocates capacity to intricate regions, saves ~10% of tokens on ImageNet at comparable fidelity
(~9% zero-shot on COCO), gives tree causality for free, and a 947M GPT-style model reaches 2.08 gFID
at 256² with zero-shot spatial control.

**(c) If you train the VQ instead of using a frozen one**, two recipes are non-negotiable:
- 2609.26774 **StableVQ**: dynamic STE (fixes the encoder's objective under low utilisation), region
  VQ loss (the codebook tracks the encoder distribution independently), decoupled LR schedules for
  encoder/decoder vs codebook. No new parameters; consistent gains in stability, utilisation and
  reconstruction across codebook sizes.
- 2605.06870 **AE warm-up before VQ**: dimensional collapse (representations living in 1–2% of full
  rank) imposes a hard loss lower bound that codebook tricks cannot beat. Training as an unquantized
  AE first restores rank: VQGAN effective dimension 3–5 → 17–19 with rFID −17–35%; WavTokenizer
  dimension 4 → 17–19 with PESQ +11–14%. There is a predicted optimal warm-up length, i.e. a
  principled switch criterion.

**(d) Quantizer choice is measurable, not a matter of taste.** 2609.02107 gives a unified
rate–distortion view of VQ/PQ/SQ for visual tokenisation: minimise *distortion* at a given coding
rate rather than maximise codebook utilisation (utilisation is a symptom; the STE gradient
discrepancy is the mechanism), and only compare quantizers at **matched rate and matched latent
statistics**. Under those conditions they recover the VQ < PQ < SQ distortion hierarchy. This is the
protocol to cite when you compare your ladder to a flat baseline.

**What is implemented** (`TreeCodebookCodec` + `TreeHierarchy`): construction (c). The tree is built
by recursive 2-means over the flat codebook (split at the median of each node's members' projection
on their first principal direction), node vectors are the mean of the leaves below them, level *k*
uses tree depth `d_min + (K-1-k)` where `d_min` is the finest depth whose node count still fits the
codebook cap `V`, and ids are compacted per level so a code can never exceed `V` or hit NULL.
`DinoKmeansImageCodec` supplies the leaves from frozen DINOv2/v3 patch features + an offline k-means
codebook (see §7 for the recipe).

---

## 4. Audio

Audio's hierarchy is **temporal rate**, and your README already flags the failure mode: pooling a
spectrogram 32× destroys formants. The literature agrees and offers three upgrades.

**Default recommendation — dual-stream semantic→acoustic at a fixed low frame rate.**
- 2609.08703 shows a working system on **DualCodec at 12.5 Hz**: a semantic stream plus successive
  acoustic refinements, and it *assumes prosody is already established when the semantic stream is
  generated* — allocating one 1.7B predictor to the semantic stream (which also fixes utterance
  duration) and three 0.6B predictors to the acoustic refinements. That is a concrete, published
  argument for coarse-rung-first generation, i.e. for your level ordering; ≈200 ms to first audio on
  one RTX 5090, RTF 0.02 across 8 concurrent inputs, and listeners preferred its prosody in 70.25%
  of 400 blinded comparisons.
- 2606.06743 **HybridCodec** resolves the "distil SSL into RVQ-1" vs "keep two streams" argument by
  doing both: separate semantic and acoustic branches, SSL distilled into the semantic one so no SSL
  model is needed at inference, 3× faster than existing dual-stream models.
- Background: 2510.16841 (SAC), 2509.17006 (MBCodec), 2506.15456 (factorised RVQ-GAN).

**Variable-rate is the right answer to "levels are rates".** 2610.29434 **RubatoCodec** (built on
X-Codec-2.0) gives an *exact utterance-level event budget* with streaming decode: adaptive firing
splits speech into variable-duration events from feature/spectrum change alone (no learned boundary
predictor), each event is discrete codes + an explicit duration, and adaptive allocation beats
fixed-rate coding in reconstruction and intelligibility at low event rates — best intelligibility
among compared codecs on the evaluated English and Japanese benchmarks at comparable bitrates, with
lower WER and higher UTMOS/UTMOSv2 in a matched TTS comparison. Code and weights are released. For
your schedule this is the coherent way to make level *k* coarse in *rate* without destroying
formants: pick fewer, longer events instead of averaging mel frames.

**Two cautions that change how you word results.**
- 2610.26881 (AuRA): decoders trained on frozen SSL audio encoders generate as well as or better than
  dedicated autoencoders **despite worse reconstruction** — reconstruction fidelity is not a proxy
  for generative quality. Same conclusion from the image side (FeaTok).
- 2610.05691 (AudioGAR): a codec decoder is trained on encoder latents but deployed on
  *generator*-produced latents; the resulting reconstruction/generation gap is measurable under FD
  and FAD, and closing it costs only 1.5% of the original audio hours / 0.26% of training cost by
  fine-tuning the decoder on diffusion-denoised intermediate latents (encoder and generator frozen).
  If your eval (quality-vs-NFE) shows a floor the model cannot cross, this is a plausible cause and a
  cheap fix.

**Fidelity-only baselines to beat, and the low-bitrate frontier.** DAC (44.1 kHz, 9 RVQ stages) is
the strongest openly installable fidelity baseline and is what `DacAudioCodec` wraps; X-Codec-2.0 at
25 Hz / 24 kHz is the current multilingual reference (2601.20185). At the extreme end 2610.15453
**BridgeCodec** maps an 8 kHz source encoder to a 48 kHz decoder at **1 kbps** via a Schrödinger
bridge with a Mamba-enhanced U-Net — relevant only as evidence that decoder-side translation between
frozen codecs is possible without co-training.

**What is implemented**: `DacAudioCodec` (optional) = DAC encoder latents per frame + tree over the
RVQ-1 codebook, so the ladder exists on a real codebook today. The dual-stream / variable-rate
upgrade is deliberately **not** implemented: it changes the frame count semantics (`l_k` becomes an
event count), which is a schedule-level decision (§6).

---

## 5. Evaluation protocol this repo should report

1. **Matched rate and matched latent statistics** when comparing quantizers or ladders
   (2609.02107). Fixing `V` and `l_k` is necessary but not sufficient: also match the feature
   statistics the codebook sees, or the comparison measures the features, not the quantizer.
2. **Rate–distortion curves, not single points.** Your `eval` prints quality-vs-NFE per level; add
   the rate axis (`l_k × log2 V` bits per sample) so a coarse rung's win is a point on a curve.
3. **The constant-predictor floor stays.** `eval` already prints it (a score under it is not a
   result). With the entropy text codec the coarse text rungs are still a hash — the floor is what
   tells you whether the model beat "predict the marginal".
4. **Separate reconstruction from generation.** Report the codec's reconstruction metric and the
   *generated* quality separately, and never infer one from the other (2610.26881, 2610.22060).
5. **Per-level, per-modality.** Already in `train`/`eval`: keep it, and state the coupon-collector
   caveat — coarse rungs have `≤ 2^(d)` distinct codes when the tree is shallow, so accuracy on
   level 0 saturates early and is not evidence of learning.

---

## 6. Where the current design is still weakest (say this in the paper)

- **One axis, two meanings.** `schedule.py` binds a level to `R**(K-1-k)` in *every* modality. For
  text and image that is a spatial/semantic axis; for audio it is a temporal rate axis; the literature
  treats them separately (rate-distortion for compression, concept depth for generation). A
  **two-axis v2 ablation** — rate (frame/token count) × semantic depth (tree depth) — is what
  DLCM's compression-aware scaling law (2512.24617) and SemanTok's "each prefix must reconstruct
  DINO features on its own" (2610.00686) suggest. Not implemented; proposed.
- **Coarse rungs are not trained to be semantic.** SemanTok adds lightweight heads that reconstruct
  frozen DINO features *from each retained token prefix alone*; the tree codec here inherits whatever
  semantics the leaf codebook had. Adopting the prefix-reconstruction objective on level-0/1 codes is
  the cheapest available upgrade and does not touch the mask or the packing.
- **The text segmentation is a proxy.** Word starts + 4-gram repetition is deterministic and cheap,
  but it is not a learned entropy boundary and makes no claim about compressibility.
- **Audio still uses a rate-pooled spectrogram in the default tier**, which the README calls the weak
  link. That is the honest state: the fix is §4, not a tuning knob.

---

## 7. Operating the tier knob

```bash
python run_omni.py check           # invariants + codec ladder contract + producer selfcheck
python tests/test_codec_ladder.py  # contract, tree ancestry, text losslessness, visible fallback
```

`produce.json` carries `codec_tier`:

| tier | behaviour |
|---|---|
| `auto` (default) | learned codec if importable here, else the stand-in; the winner **and the reason** go into every shard's `meta.codecs[modality]` and into the producer log |
| `learned` | learned only; raises with the exact reason rather than degrading |
| `standin` | dependency-free placeholder — use to reproduce an existing run byte-for-byte |

Nothing silently swaps: `codec.info()` always reports `tier` and `note`, and
`run_omni.py check` prints one line per modality.

**Enabling the image codec** (offline, once; needs `transformers` and a GPU, ~1 h at 50k images):

```python
# collect patch tokens, then k-means to V centroids, save as run/codecs/image_kmeans.npz
import numpy as np, torch
from transformers import AutoImageProcessor, AutoModel
mid = "facebook/dinov2-base"
proc, net = AutoImageProcessor.from_pretrained(mid), AutoModel.from_pretrained(mid).eval()
feats = []                                   # ~50k images, ~64 random samples each
# for img in images: feats.append(net(**proc(images=img, return_tensors="pt")).last_hidden_state[0, 1:])
X = np.concatenate(feats).astype("float32")
C = X[np.random.default_rng(0).choice(len(X), 1024, replace=False)].copy()   # k-means++ / Lloyd here
np.savez("run/codecs/image_kmeans.npz", centroids=C)
```

Set `NCM_IMAGE_CODEBOOK` to the file path elsewhere, `NCM_DINO_MODEL` to change the backbone.
`pip install descript-audio-codec` enables the audio codec; `NCM_DAC_MODEL` selects the DAC variant.

---

## 8. References (arXiv ids)

**Text** 2610.05978 (byte LMs: scaling, emergent abstractions, information allocation) ·
2512.24617 (Dynamic Large Concept Models) · 2610.26695 (hierarchical token merging) ·
2610.26225 (BPE budget as model selection) · 2610.10304 (SemanticFold: no single compression
certificate).

**Image** 2610.22060 (FeaTok) · 2609.03641 (Tree-VQ) · 2610.10497 (QuadTok) · 2609.26774 (StableVQ) ·
2605.06870 (AE warm-up before VQ) · 2609.02107 (unified rate–distortion view of VQ/PQ/SQ) ·
2609.39451 (ResARC, residual-aware progressive coding) · 2610.12307 (BudgetPix, compute-adaptive
tokenisation) · 2610.13322 (CoVA) for continuous-VAE composition.

**Audio** 2609.08703 (hierarchical codec TTS on DualCodec 12.5 Hz) · 2606.06743 (HybridCodec) ·
2610.29434 (RubatoCodec, budgeted variable-rate) · 2610.26881 (AuRA: SSL-space audio latents) ·
2610.05691 (AudioGAR: decoder train/generation mismatch) · 2601.20185 (X-Codec-2.0, 25 Hz) ·
2610.15453 (BridgeCodec, 1 kbps) · 2605.06870 (WavTokenizer dimensional collapse).

**Cross-modal / hierarchy** 2610.00686 (SemanTok: coarse prefixes must carry semantics) ·
2606.30811 (AVTok: one unified 1-D audio-video tokenization) · 2607.20253 (hierarchical
autoregressive planning + flow-matching rendering for full songs).
