# Feature-codec notebook (`TOKENIZER="feat"`)

**Status: written, not executed.** Nothing here was run on the machine that wrote it. Expect to fix small errors on the first Colab run;
the codec checks (section 7d) are there so that a bad codec fails before any model compute.

## What changed against the previous notebook

| file | change |
|---|---|
| `src/06b_feat_codecs.py` (new) | `DinoImageCodec` (DINOv2 patch tokens), `DacAudioFeatureCodec` (DAC encoder latents), `TextKMCodec` / `TextEntropyCodec`, `TreeHierarchy`, `fit_feature_codecs()` |
| `src/07d_codec_checks.py` (new) | contract + usage + learnability checks and a gate, before the model is built |
| `src/12b_learned.py` (new) | "did it learn" cells after the verdict |
| `src/06_codecs.py` | `Codecs` builds the feature codecs for `TOKENIZER="feat"`; `state()/load()` also carry the text codec |
| `src/01_setup.py` | `TOKENIZER` default is now `"feat"` (`"vq"` and `"kmeans"` still work) |
| `src/07c_stage.py` | fit branch for `feat`; resume check knows the `feat` kind; shards are refused with `feat` |
| `src/10_correct.py` | codec round trip no longer assumes a pixel / mel decoder |
| `src/13_generate.py` | image generation is skipped for a codec without a pixel decoder |
| `run_src.py` | cell order |

## Build and run (do not run this on a laptop; build it where you train)

```bash
pip install -q nbformat                       # build_nb.py needs it
python build_nb.py --out NCM_MoE_Omni_Feat.ipynb
```

First run: `NCM_TIER=T0 NCM_TOKENS=0.05`, with `NCM_CKPT_DIR` set to a **new** folder (a checkpoint from another codec is refused).
The codec fit is cached at `$CKPT_DIR/feat_codecs.pt` (or `NCM_TOK_CACHE`), so later sessions skip it.

`pip install transformers descript-audio-codec` is run automatically when `INSTALL` is on. `descript-audio-codec` pulls
`descript-audiotools`, which can pin older `protobuf`; if another package breaks after it, restart the runtime. The DINOv2 and DAC
weights are downloaded from Hugging Face / GitHub on the first run.

## Knobs (env `NCM_<NAME>`)

| knob | default | meaning |
|---|---|---|
| `TOKENIZER` | `feat` | `feat` / `vq` / `kmeans` |
| `FEAT_QUANT` | `tree` | `tree`: 256 k-means leaves + a balanced tree, level k = ancestor; `perlevel`: independent 256-code k-means per level |
| `TEXT_CODEC` | `km` | `km` data-fit byte-embedding ladder / `entropy` word-aligned hashed spans (omni's) / `hash` (the original) |
| `DINO_MODEL`, `DAC_MODEL` | `facebook/dinov2-base`, `44khz` | backbones |
| `FEAT_IMAGES`, `FEAT_AUDIO`, `FEAT_TEXT` | 6000, 2000, 8000 | rows streamed once to fit the codebooks |
| `CODEC_CHECK` | on | run section 7d |
| `CODEC_GATE` | `stop` | `stop` raises on a hard failure; `warn` prints and continues |
| `CHK_IMAGES`, `CHK_AUDIO`, `CHK_TEXT`, `CHK_PROBE_STEPS` | 1200, 500, 3000, 300 | size of the check sample and of the probe |

## What the checks mean

Hard (stop the run): wrong level length, a code outside 1..256, a constant level, a non-finite metric, a text finest level that does
not decode to the input, or a codec that raises.

Soft (printed): low finest-level entropy, a level using few of its codes, low image feature R^2 / audio mel R^2, a finest level whose
probe gain from its parent level is under 0.05 bits.

The **probe** is the number to read: a small conv net predicts level k from level k-1 on held-out rows. If it gains nothing, the
big model cannot gain much at that level from the previous one either. `12b_learned` prints the model's gain next to it.

## Known limits

- **No pixel decoder for the DINO image codec.** There is no image generation and no PSNR; the image checks use feature R^2. Use
  `TOKENIZER=vq` if you need to *see* images.
- **Audio decodes through DAC** from prototype latents, resampled to 16 kHz; the 44.1 kHz encoder frames (706) are pooled to the
  notebook's 512 frames and linearly interpolated back for decoding. Expect imperfect audio.
- **No shards** with `feat` (the stage cell refuses it).
- **Text coarse levels stay weakly predictable** with every text option measured so far (hash 7-18% of the entropy explained by the
  parent code, the `km` ladder 8-35%); the finest level is the exact byte.
- `TextEntropyCodec` is a python loop per row (about 0.1 s per micro-batch of 128 rows); fine, not fast.
- The DINO / DAC forward passes run inside the data prefetch thread; if `data-wait` is high, that is the first thing to look at.
