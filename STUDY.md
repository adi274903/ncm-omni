# Training a 1.68B sparse MoE on one GPU: 170 TFLOPS, what broke, and what I'd change

A training study, not a model release. NCM-MoE Omni is a from-scratch sparse mixture-of-experts that predicts
text, image and audio as levels of discrete codes, coarse to fine. Everything below comes from
[`ncm_moe_nb/NCM_PROJECT_REPORT.html`](ncm_moe_nb/NCM_PROJECT_REPORT.html), where each number is marked
MEASURED (from a real run), ESTIMATE (arithmetic on measured numbers) or UNVERIFIED (reasoning not yet tested).
Status: work in progress. Two of three modalities learned; audio overfit. No weights are published.

## 1. Setup

| | |
|---|---|
| Parameters | 1.68B total, about 364M active per token |
| Trunk | d=1536, 16 layers, GQA 16/4 [6], QK-norm [7], RMSNorm [9], block-causal attention over levels |
| Experts | 3 modality groups x 2 SwiGLU [8] experts = 6 experts (about 264M each); sparse MoE [3, 4], group forced by modality, expert-choice routing [5] inside the group |
| Levels | K=5, ratio 4, predicted coarse to fine [15, 14]. Text and image 1/4/16/64/256 slots, audio 2/8/32/128/512 |
| Vocabulary | 257 per level: 256 codes + NULL |
| Hardware | one NVIDIA RTX PRO 6000 (95 GB) |
| Main run (T1) | 2B codes, 2,006 steps, about 65k codes/s, about 170 TFLOPS sustained, data-wait 0% |

## 2. Throughput and MFU

| Setting (RTX PRO 6000) | codes/s | TFLOPS |
|---|---|---|
| T1 run, feature codecs, 16 kHz audio | 65k | ~170 |
| Best real training (levels attention, flat MoE dispatch, micro-batch 49152, no activation checkpointing) | 72-73k | 187-190 |
| Synthetic compute benchmark, micro 32768 / 49152 | 75.9k / 77.4k | 193.3 / 197.1 |
| `max-autotune-no-cudagraphs` | 78.4k | 199.7 (+1.3%, about 20 min compile) |

**MFU [11] is about 34% for the T1 run (170 / 500) and about 38% at the best setting (190 / 500).** The 500 TFLOPS
dense bf16 peak for the RTX PRO 6000 is the value in `src/02_hardware.py`, written from memory and not checked
against NVIDIA's datasheet, so treat MFU as approximate.

Where the time went at the best setting (profiled): matmul about 50%, copies and elementwise about 30% (RMSNorm
backward, dtype casts), attention about 9%, SwiGLU about 5%. Peak memory about 69 GiB.
FP8 was the next large lever and was declined because of the accuracy risk.
The notebook projects wall-clock time for larger runs with the 6 x active parameters x tokens rule [12].

## 3. Audio tokenizer and vocabulary decisions

- **Codes come from frozen features, not raw signal.** Audio uses DAC [1] encoder latents, images DINOv2-base [2]
  patch features; both are pooled up the 5-level ladder and quantized (in the spirit of VQ [10]) through a tree
  hierarchy over 256 k-means leaves.
  Text uses a data-fit byte-embedding ladder.
- **Vocabulary 257, codes 1-based, 0 = NULL.** Every codec emits 1..256, so a real code can never be confused with
  padding. Audio needed this most: its codes are argmax indices, which start at 0. Absent modalities are NULL,
  not dropped, so the batch shape stays static.
- **16 kHz instead of 44.1 kHz DAC.** At 44.1 kHz the audio encoder cost more than the audio model step itself:

| Per micro-batch | 44.1 kHz DAC (fp32) | 16 kHz DAC (fp32) | 16 kHz DAC (bf16) |
|---|---|---|---|
| Audio encode, s | 0.690 | 0.262 | 0.200 |
| Audio model forward+backward, s | 0.553 | 0.553 | 0.553 |
| Throughput | 55.8k codes/s, 145 TFLOPS | 65.3k, 170 TFLOPS | not run |

- **fp32 encode kept.** bf16 kept 98-99% of codes identical at every level but would add only about 2% throughput.
- **Encode in chunks of 8 clips.** Encoding a whole micro-batch at 44.1 kHz in fp32 ran out of memory at step 2.
- Audio codes were the most learnable from the parent level (44.1 kHz check: log-mel R² 0.59, probe gain
  +0.69 / +0.44 / +1.39 / +2.07 bits at L1-L4).

## 4. Results (held-out CE, nats, mean over levels)

| Eval | Text | Image | Audio |
|---|---|---|---|
| Step 753 (0.73B codes) | 3.49 | 1.90 | 1.52 |
| End of T1 (2B codes) | 3.34 | 1.76 | 1.78 |
| Constant-predictor baseline | 4.90 | 4.12 | 2.52 |

Every level of every modality beats its baseline at the end. Shuffling the prompt or caption raises text and
image CE at the coarse levels, so both use their conditioning. Replacing the previous level with random codes
raises L4 CE by up to 6.5 (text), 4.5 (image) and 6.3 (audio) nats, so the model reads the earlier levels.

## 5. What broke

- **Audio overfit.** Training CE 1.001 against held-out 1.775 (gap +0.774); text (-0.104) and image (+0.058) show
  no gap. Held-out audio CE fell to 1.52 at step 753, then rose to 1.78 by the end. The checkpoint was overwritten
  every 30 minutes, so the best audio checkpoint is lost. The cause (too few distinct audio clips, source
  weights, clip filtering) was not diagnosed.
- **An early model learned to copy its target.** Each position was fed its own target code, which sits in the
  residual stream, so train loss reached 7.7e-09 in 2k steps while generation scored 0.122, below the 0.174
  constant baseline. The input is now shifted so no position can read its own target. That run is void.
- **The data feed, not the GPU, was the bottleneck on an H100.** Sources with fewer Parquet shards than workers
  fell back to a row filter, so every worker re-read the whole dataset: 16 workers gave 39k codes/s with 68%
  data-wait. 12 workers removed the stall; the H100 still showed 34-50% data-wait, and its compute sweep never
  ran (the pod balance ran out).
- **CUDA graphs failed with gradient accumulation** and were dropped.
- **Text coarse levels stay close to noise** with every text codec tried, and generated text is still babble.
  Generation quality was not shown for any modality.

## 6. Single-factor comparisons (ablations)

| Change | Result |
|---|---|
| Audio codec 44.1 kHz to 16 kHz | 145 to 170 TFLOPS, audio encode 0.690 to 0.262 s per micro-batch |
| DAC encode fp32 to bf16 | 98-99% identical codes, about +2% throughput (not adopted) |
| Plain masked SDPA instead of levels attention | about 13% slower in a real run |
| `max-autotune-no-cudagraphs` | +1.3% for about 20 min of compile |
| Data workers: 16 (row-filter path) to 12 (shard split) | 39k to 77.8k codes/s on the H100, data-wait 68% to 34% |
| Image codec: k-means to learned residual VQ-VAE | +1.93 dB finest-level PSNR (21.7 dB) |
| Text codec: learned VQ-VAE | coarse codes near-uniform noise; not adopted |

Not yet done: a controlled run that changes only the codec. Cross-codec CE values are not comparable.

## 7. What I'd change with 8 GPUs (a plan, not a result)

- **Fix audio before scaling.** More distinct audio clips or a different source mix, stronger regularization,
  and keep the best held-out checkpoint instead of overwriting one file.
- **Take codec encoding out of the training step.** Pre-encode codes offline, so eight GPUs do not each pay the
  audio encode cost and the data feed stops being the limit.
- **Data parallel first.** The model fits on one 95 GB card (about 69 GiB peak), so plain data parallelism across
  eight GPUs is the simplest scale-up; sharding (FSDP [13]) only becomes necessary if the experts grow.
- **Run the controlled codec ablation and the H100 compute sweep** that this budget could not cover.
- **Try semantic coarse codes for text,** since every text codec tried left the coarse levels near noise.

## Files

- `ncm_moe_nb/src/` - the notebook source cells; `build_nb.py` assembles the notebook (order in `run_src.py`).
- `ncm_moe_nb/NCM_PROJECT_REPORT.html` - the full report with every measured table.
- `ncm_moe_nb/RUNPOD.md`, `ncm_moe_nb/FEAT_CODECS.md` - pod and codec runbooks.
- The rest of the repository is the earlier NCM Omni package this work grew from (see `README.md`).

## References

Each entry was checked against its arXiv page on 2026-11-15. They are background for the methods used, not
sources of any number in this study.

1. Kumar et al. *High-Fidelity Audio Compression with Improved RVQGAN* (DAC). 2023. [arXiv:2306.06546](https://arxiv.org/abs/2306.06546)
2. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
3. Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
4. Fedus et al. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. 2021. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
5. Zhou et al. *Mixture-of-Experts with Expert Choice Routing*. 2022. [arXiv:2202.09368](https://arxiv.org/abs/2202.09368)
6. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
7. Henry et al. *Query-Key Normalization for Transformers*. 2020. [arXiv:2010.04245](https://arxiv.org/abs/2010.04245)
8. Shazeer. *GLU Variants Improve Transformer* (SwiGLU). 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
9. Zhang and Sennrich. *Root Mean Square Layer Normalization*. 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
10. van den Oord et al. *Neural Discrete Representation Learning* (VQ-VAE). 2017. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
11. Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways* (defines model FLOPs utilization). 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
12. Kaplan et al. *Scaling Laws for Neural Language Models*. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
13. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*. 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
14. Tian et al. *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*. 2024. [arXiv:2404.02905](https://arxiv.org/abs/2404.02905)
15. Kumbong et al. *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*. 2026. [arXiv:2610.14162](https://arxiv.org/abs/2610.14162)
