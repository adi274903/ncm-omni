# %% [markdown]
# # NCM-MoE v2: 16,384-bit images (32x32 grid, two codebooks) + DAC audio + km text
#
# Same model and training code as the main notebook, with three changes (all from the codec-budget tests):
#
# | | v1 (previous T1 run) | **v2 (this notebook)** |
# |---|---|---|
# | image codec | frozen DINOv2 features, 16x16 grid, **no pixel decoder** | learned multi-scale residual VQ-VAE, **has a pixel decoder** |
# | image bits at the finest levels | 256 codes x 8 = 2,048 bits | 2 x 1,024 codes x 8 = **16,384 bits** |
# | image ladder (slots per level) | 1 / 4 / 16 / 64 / 256 | 4 / 16 / 64 / 256 / 1,024 / 1,024 (the last level is the second codebook, conditioned on the first) |
# | live codes per image row | 341 | **2,388** (about 7x: the model sees about 7x fewer images per step at the same compute) |
# | audio codec | DAC 16 kHz latents, k-means tree, 512 slots x 8 bits = 4,096 bits/clip (~500 bit/s, about 1 DAC codebook) | same DAC latents, **learned residual quantiser, 8 levels at the finest rate = 32,768 bits/clip (~4 kbps = 8 DAC codebooks)** |
# | live codes per audio row | 680 | **about 4,266** (about 6x) |
# | text codec | `km`: PPMI-SVD byte embeddings + k-means for the coarse levels, exact byte at the finest | **learned multiscale residual VQ-VAE** for the coarse levels (the NCM paper's tokenizer idea, trained with a random-prefix loss); the finest level is still the exact byte |
# | run size | T1 | T1, `TOKENS=0.3`, pinned to **301 steps** with `NCM_STEPS=301` (the default here; the longer image and audio rows would otherwise shift the step count a little) |
#
# **What to read in the output**
# 1. *Codec fit:* `image PSNR by level`. The 16x16 grid reached 21.8 dB and the 32x32 grid with one codebook about 24.9 dB in the
#    budget test; the last level here (both codebooks) should be higher. Code perplexity per level should stay well above 100 (256 = all codes used).
# 2. *Codec checks (7d):* the probe gain of the last two image levels. A last level with a gain near 0 is noise the model cannot learn.
# 3. *Training:* image per-level CE. Images now decode to pixels, so the generate cell shows real images, level by level.
#
# The old checkpoint is not reusable (different image layout and codec), so this notebook uses its own `CKPT_DIR`.

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive not mounted (not on Colab?):", e)

%env NCM_IMAGE_RES=v2
%env NCM_IMAGE_CODEC=vq
%env NCM_VQ_IMAGES=16384
%env NCM_TOKENIZER=feat
%env NCM_AUDIO_RES=v2
%env NCM_TEXT_CODEC=vq
%env NCM_TIER=T1
%env NCM_TOKENS=0.3
%env NCM_STEPS=301
%env NCM_CKPT_DIR=/content/drive/MyDrive/ncm_feat_v2_t1
%env NCM_DAC_MODEL=16khz
%env NCM_CODEC_GATE=warn
%env NCM_WORKERS=12
