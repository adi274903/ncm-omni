# NCM-MoE: a from-scratch multimodal Next-Concept-Modelling MoE (text / image / audio)

Hand-off document for another model or engineer. Everything here comes from the working session; numbers marked
**(est.)** are estimates that were never measured on the stated hardware. Nothing was benchmarked on an H100 or B200.

## 1. What this is

A self-contained Colab notebook (built from `src/*.py`, nothing imported from the old `ncm_omni_v2` repo) that trains a
small, **text-conditioned multimodal generator**. It is *not* an "omni" model in the any-input/any-output sense (see 10).

- One transformer predicts **every code of a level in one forward**, coarse to fine; generation = K forwards (K = 5 for the
  full preset) instead of one per token. This is the Next-Concept-Modelling (NCM) ladder idea.
- Text, image and audio share one level ladder (same compression ratio per level), a shared trunk, and modality-specific experts.
- Capabilities by design: text to text (prompt to response), text to image (caption to image), text to speech (transcript to audio, Griffin-Lim vocoder).
- Data is streamed straight from the Hugging Face Hub (RAM-only cache); only checkpoints (and optionally code shards) hit disk.

## 2. Architecture

**Level ladder.** K levels, ratio R = 4. Level k of a modality with finest length n has `n / R^(K-1-k)` slots. Images use a 2D grid
(2x2 pooling per level); text and audio are 1D. Codes are 1..256, 0 = NULL, vocab 257.
**Packing.** A row = `[text prefix | level 0 | level 1 | ... | level K-1]`. Prefix = prompt / caption / transcript bytes (cond = 128).
Attention is block-causal over levels (a slot sees the prefix and all slots of levels <= its own, bidirectional inside a level).
**No leak.** The input at level k is the *upsampled codes of level k-1* (level 0 reads NULL), never its own targets. An earlier idea leaked
its targets and scored below the constant baseline; the correctness cell asserts this is gone.

**Row types (homogeneous, 4 static shapes):** `text0` (plain text), `text` (prompt to response), `image` (caption to image), `audio` (transcript to speech).
Full preset sequence lengths T (target Tt): text0 341 (341), text 469 (341), image 469 (341), audio 810 (682).

**Model (preset `full`).** d=1536, L=16, GQA 16/4, QK RMSNorm, learned absolute positions, `F.rms_norm`.
Experts: 3 modality groups x 2 SwiGLU experts (h=3584, about 264M per expert). Forced routing by modality tag; inside a group,
**expert-choice per level** (each expert takes its top ceil(cf*l) tokens, cf=0.5, so ~1 expert/token, 0/1/2 possible, static shapes),
both experts run as one `bmm`. Output heads are zero-initialised (init loss = ln 257 = 5.549). Loss: chunked, checkpointed CE + z-loss 1e-4.
**Counts:** 1,684.7M total (printed by the notebook), ~360M active per token (trunk ~100M + one expert); `tiny` preset (d=512, L=8) = 147M total, ~29M active.
Specs: full K=5, text 256, image 256 (16x16 patches of 16 px), audio 512 frames; tiny K=4, text 128, image 64, audio 512.

## 3. Frozen codecs (the main quality ceiling)

- **Text:** the raw UTF-8 byte (+1) is the finest level (lossless). Coarse levels = sign bits of an average-pooled random 8-plane projection of bytes (SimHash-style, 8 bits to a code). No tokenizer, no BPE.
- **Image:** patch pyramid (avg-pool per level) + per-level k-means VQ (256 centroids) on raw 16x16x3 patches; fit from 4,096 streamed images (full). Decoder = codeword tiles. Output is blotchy by construction.
- **Audio:** 16 kHz log-mel (hop 256, 80 mels), time-pooling pyramid, per-level k-means (256); decode = pinv to magnitude + Griffin-Lim. Fit from 1,024 clips.
- Codebooks are fit at startup from streamed samples held in RAM, stored only in the checkpoint. Changing a codec invalidates all weights and any stored shards.

## 4. Data pipeline

Sources (weights inside a row type): text0 fineweb-edu (sample-10BT), wikipedia (20231101.en), cosmopedia-v2, codeparrot-clean, open-web-math;
text NuminaMath-CoT, smoltalk; image cc12m-wds, cc3m-wds, DocVQA, COCO-Caption2017; audio librispeech (clean), libritts_r, peoples_speech.
Code share per modality: text 40% (15% of it prompt-response), image 35%, audio 25%. Rows are converted to row-type probabilities in `type_probs()`.
Pipeline: forked worker processes (`RowStream`, arrival-order queue, not a DataLoader) -> micro-batch builder -> GPU codecs on a side CUDA stream
(nvJPEG for JPEG) -> RAM ring (`Prefetcher`) -> trainer. Partitions: 0 = held-out, 1 = codec fit, 1..W = training workers. HF cache lives on `/dev/shm`.
**Known issues:** (a) HF API rate limit, HTTP 429 "1000 requests / 300 s window": too many workers opening all sources at once, plus retry storms. Use WORKERS <= 12 (8 for a cold start), have `HF_TOKEN` set.
Suggested (not yet applied): stagger worker starts and sleep ~90 s on a 429 without counting it as a failure. (b) Small sources (smoltalk 9 files, numina 5, coco 2) are partitioned by row index, so every worker reads them whole. (c) Audio sources are small and repeat heavily (`libri` reopens every ~20-30 steps); the held-out audio gain is the number to trust.

## 5. Training setup (notebook knobs)

`PRESET, TIER (T0 0.02B / T1 2B / T2 20B / T3 50B live codes), TOKENS, WORKERS, ACT_CKPT, WARMUP, EVAL_EVERY_B, CKPT_DIR, CKPT_MIN, SESSION_HOURS`
(each overridable by an `NCM_<NAME>` env var). Optimiser: fused AdamW (0.9, 0.95), wd 0.1 on matrices, LR 6e-4 (full), linear warmup then cosine to 10%, grad-clip 1.0, bf16 autocast.
~1M live codes per optimizer step (gradient accumulation over micro-batches of 32,768 packed positions, one row type each). Early abort at step 200 if loss fell < 5%.
Evaluation: teacher-forced held-out CE and top-1 per (modality, level) against the **training-unigram constant predictor** (add-one smoothed); the verdict cell prints PASS / PARTIAL / NOT LEARNING.
Checkpoints (weights, optimizer, step, seen, codebooks) to `CKPT_DIR`; a watchdog saves and stops before `SESSION_HOURS` (11.5 on Colab); resume re-streams with a fresh shuffle seed (approximate resume).
Kernels: Triton fused SwiGLU as a `torch.library` custom op (the startup probe picks torch.compile when equal: 0.23 vs 0.22 ms), chunked/checkpointed CE, flex-attention (skipped on GPUs with < 160 KB shared memory, falls back to SDPA), `torch.compile`.

## 6. Results so far (all on streamed data)

| run | hardware | data seen | held-out CE vs training-unigram baseline | notes |
|---|---|---|---|---|
| T0, tiny (29M active) | A100 | 0.02B, 579 steps | gains over baseline: text 0.39, image 0.81, audio 0.96 nats | verdict PASS (30.5% loss drop); samples garbled, as expected |
| pilot, full (360M active) | A100 80GB | 0.3B, 301 steps | step 250: text 2.57 (3.01), image 4.45 (5.18), audio 2.99 (3.89); acc 37/12/43% | loss 5.55 -> ~2.6 |
| T1, full, in progress | Colab RTX PRO 6000 (96 GB) | 2B, 2,010 steps | step 150: text 2.79 (3.07), image 4.76 (5.19), audio 3.13 (3.78); loss 2.81 at step 170 | tracks the pilot; ETA ~8.7 h |

Throughput (full preset, measured): A100 80GB 47k codes/s in training (benchmark 52k, MFU ~39-42%, peak memory 43.9 GiB with ACT_CKPT=0);
RTX PRO 6000 62k codes/s, 161 TFLOPS, data-wait 0%. Colab burn rates: A100 6.77 units/h, RTX PRO 6000 8.9 units/h.
Generation was only tried on the T0/pilot models: text is byte-level noise with English texture, images blurry blobs, speech garbled (Griffin-Lim on a coarse codec).

## 7. Bugs found and fixed (history)

- Streaming stall with a torch DataLoader (strict round-robin) -> replaced by `RowStream`.
- Init loss 5.706 vs ln 257 -> zero-init output heads. Isolation test saw zero gradients with zero heads -> the test temporarily swaps in a random head.
- `KeyError 'loss'` in the early-abort (val records in HIST) and an unfair held-out baseline (entropy of a tiny val set) -> training-unigram baseline.
- Chunked-CE test failed under TF32 -> TF32 off and a float64 reference in that test.
- flex-attention compile failed on shared memory (head_dim 96 default tiles 178 KB > A100 163 KB; RTX PRO 6000 has 99 KB) -> smaller `kernel_options`, a backward/all-geometry probe, and a shared-memory check that falls back to SDPA.
- Storage check false positives; `pkill -f` killed its own shell.

## 8. Cost / time estimates (est. unless noted)

Compute 6*N_active*codes: A100 47k codes/s -> 2B ~12.4 h, 10B ~59 h, 20B ~118 h (4.9 d), 50B ~295 h. H100 SXM (est. ~125k codes/s, needs ~32 vCPU feed): 10B ~22 h.
The feed is the next bottleneck: ~4-5k codes/s per worker (12 workers: 47-67k), so 16 vCPU holds an H100 to ~60-85k. RunPod listed prices seen: A100 SXM $1.79/h, H100 SXM $3.99/h, RTX PRO 6000 $2.49/h (all 32 vCPU).
Colab units cost ~$0.10 each on standard plans (Google AI Ultra at $100 for 1,000 units is the same per unit).

## 9. Shards experiment (built, then abandoned)

Implemented `SHARD_MODE` = `record` / `read` / `encode` (encoded `uint8` code shards per row type, codebook fingerprint, seeded resumable reader, fit/val caches, tar step) in `src/07b_shards.py`, `07c_stage.py`, `08b_encode.py`.
Conclusion: encoding is the same single pass over the Hub as training, so for one run on one GPU it only costs time (50B codes ~ 200-300 h). Only worthwhile for multi-GPU, repeated runs, or exact-resume comparisons. The notebook currently being run is the non-shard `NCM_MoE_Omni.ipynb`.

## 10. Honest limits and what to do next

- "Omni" is a stretch: only text is understood as input. No image-to-text, speech-to-text, interleaving or video. Adding image/audio-prefixed row types would be the path.
- Quality is bounded by the codecs: byte tokenizer + hashed coarse text levels, k-means image codes, mel k-means + Griffin-Lim. A learned multi-scale VQ tokenizer (VAR-style images, SNAC-style audio; names from memory, unverified) is the highest-value upgrade; it forces a re-encode and retrain.
- Image CE is high mostly because its codes are near-uniform (baseline 5.18 of a 5.55 maximum); relative gain over baseline is similar to text.
- Missing for a research claim: an autoregressive baseline of the same size on the same data, caption/prompt-use test (CE with the real vs a blank prefix), downstream metrics (bits per byte, CLIP/FID, WER), a quality-vs-NFE curve with refinement passes, multi-GPU (DDP) code.
- **Pending edit:** two-tier checkpoints: local `/content` every 15 min (`LOCAL_CKPT_DIR`, `LOCAL_CKPT_MIN`) and Drive every 2 h (`CKPT_MIN=120`), resume from the newest of the two. Patch text was written but not applied.
- Check that a `checkpoint ->` line appears in the running T1 job (none had been seen by step 170).

## 11. Files

`src/00_dotenv.py ... 14_storage.py` are percent-format cells; `run_src.py` defines `ORDER`; `build_nb.py` builds the notebook with `nbformat` validation
(`../.venv/bin/python build_nb.py` from this folder). Built notebooks: `NCM_MoE_Omni.ipynb` (streaming), `NCM_MoE_Omni_Shards.ipynb` (shard variant, in `~/Downloads` as `NCM_MoE_Omni_Shards_encode.ipynb`).
Original plan: `/home/adi274903/.claude/plans/linear-brewing-crown.md`.

## 12. Working agreements with the user

Do not run anything on the user's machine unless they explicitly ask for that specific action (they have forbidden local tests and builds by default); browser tools only when asked;
never enter or request credentials (HF token goes in a Colab secret or `.env`); report estimates as estimates and quote measured numbers separately.
