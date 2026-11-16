# %% [markdown]
# # NCM-MoE: evaluation only
#
# Restores a trained checkpoint and runs **only** evaluation: no training, no benchmark. Everything it needs is in
# `CKPT_DIR/ckpt.pt`: the model weights, the fitted codecs (DINO + DAC + text ladder), and the training history (including every
# held-out eval made during training). The held-out set and the training unigram (the "constant predictor") are rebuilt from the
# stream, since they are not in the checkpoint.
#
# Run the cells in order. What you get:
# 1. the held-out trajectory saved in the checkpoint (where did each modality peak?),
# 2. the per-level held-out table, prefix test, coarser-level test, train/held-out gap (same cell as the training notebook),
# 3. text bits per byte, 4. per-source train-versus-held-out CE (finds which audio source is memorised),
# 5. **audio through the DAC codec**: original, real clip through the codec (the ceiling), and generated audio decoded by the DAC decoder, with WER from an ASR model,
# 6. generated text samples, and the image checks.
#
# **First cell:** set `NCM_CKPT_DIR` to the training run's folder, and keep `NCM_DAC_MODEL` and `NCM_TIER` the same as in training
# (the T1 run used the 16 kHz DAC model and tier T1).

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive not mounted (not on Colab?):", e)

%env NCM_EVAL_ONLY=1
%env NCM_CKPT_DIR=/content/drive/MyDrive/ncm_feat_t1
%env NCM_TIER=T1
%env NCM_DAC_MODEL=16khz
%env NCM_COMPILE=0
%env NCM_CODEC_CHECK=0
%env NCM_CODEC_GATE=warn
%env NCM_WORKERS=12
