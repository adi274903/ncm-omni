# %% [markdown]
# ## E3. Audio through the DAC codec
#
# For held-out clips (never trained on) this plays and scores four versions of the same clip:
#
# | version | what it tells you |
# |---|---|
# | original | the raw waveform (the reference) |
# | real clip through the codec | the clip encoded to the 5 levels of DAC codes and decoded by the DAC decoder: **the ceiling**. The model cannot beat this. |
# | generated from the transcript | the model samples all 5 levels, the DAC decoder turns them into a waveform |
# | generated, coarse levels only | decode levels 0..2 / 0..3 of the generated codes: what each level adds |
#
# With an ASR model (Whisper) the word error rate of each version is computed. `WER(real through codec)` is what the **codec alone**
# costs; `WER(generated) - WER(real through codec)` is what the **model** adds. Decoding is by the DAC decoder, not Griffin-Lim.

# %%
import re
from IPython.display import Audio, display

N_AUDIO_EVAL = _knob("N_AUDIO_EVAL", 8)
RUN_ASR = _knob("RUN_ASR", True)
ASR_MODEL = _knob("ASR_MODEL", "openai/whisper-small")
GEN_TEMP = _knob("GEN_TEMP", 0.9)


def _wer(ref, hyp):
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def _words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def _play(label, w):
    x = w[0].detach().float().cpu().numpy()
    print("   ", label)
    display(Audio(x / max(1e-6, float(abs(x).max())) * 0.9, rate=SPEC.sr))


asr = None
if RUN_ASR:
    try:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=0 if IS_CUDA else -1)
    except Exception as e:
        print("ASR unavailable, so no WER (the audio is still played):", type(e).__name__, str(e)[:120])


def _transcribe(w):
    x = w[0].detach().float().cpu().numpy().astype("float32")
    return asr({"raw": x, "sampling_rate": SPEC.sr}, generate_kwargs={"language": "english", "task": "transcribe"})["text"]


if not META_ONLY:
    Kn = LAYOUTS["audio"].K                                                # includes the residual levels when AUDIO_RES="v2"
    helper = AudioCodec(SPEC, dev)                                         # only its STFT / mel filterbank, for the log-mel error
    clips = collect_rows(("audio",), {"audio": N_AUDIO_EVAL}, 0, PARTS, 1, seed=12345)    # part 0 = held-out
    wer, rmse = {"raw": [], "codec": [], "generated": []}, []
    for i, row in enumerate(clips):
        ref = bytes(row["cond"].tolist()).rstrip(b"\x00").decode("utf-8", "ignore").strip()
        raw = torch.from_numpy(row["wav"]).to(dev).float()[None] / 32768.0
        with torch.no_grad():
            via = codecs.audio.decode(codecs.audio.encode(raw), level=Kn - 1).float()       # real clip, quantised and decoded by DAC
        codes = generate("audio", ref, n=1, temperature=GEN_TEMP, top_k=64)
        with torch.no_grad():
            gen = {k: codecs.audio.decode(codes, level=k).float() for k in (2, 3, Kn - 1)}
            rmse.append(float(((helper.logmel(via) - helper.logmel(raw)) ** 2).mean().sqrt()))
        print(f"\n[{i}] transcript: {ref!r}")
        _play("original", raw); _play("real clip through the DAC codec (ceiling)", via)
        _play("generated, levels 0..2 only", gen[2]); _play("generated, levels 0..3", gen[3]); _play("generated, all levels", gen[Kn - 1])
        if asr is not None:
            h = {"raw": _transcribe(raw), "codec": _transcribe(via), "generated": _transcribe(gen[Kn - 1])}
            for k_, t_ in h.items():
                wer[k_].append(_wer(ref, t_))
            print("    ASR  raw:", repr(h["raw"][:90]), "\n         codec:", repr(h["codec"][:90]), "\n         generated:", repr(h["generated"][:90]))
    print(f"\nlog-mel RMSE, real clip through the codec vs the original: {np.mean(rmse):.3f} (natural-log mel units) over {len(rmse)} clips")
    if asr is not None:
        print("mean WER over", len(clips), "clips | raw audio:", f"{np.mean(wer['raw']):.3f}", "| real through the DAC codec (ceiling):", f"{np.mean(wer['codec']):.3f}",
              "| generated:", f"{np.mean(wer['generated']):.3f}")
        print("read: raw is the ASR's own error; codec - raw is what the DAC codec costs; generated - codec is what the model adds. WER can exceed 1.")
        del asr
        gc.collect(); torch.cuda.empty_cache() if IS_CUDA else None
