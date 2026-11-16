# %% [markdown]
# ## E5. Text likelihood in bits per byte
#
# The model predicts all codes of a level in parallel, so a level's CE is conditioned on the **true** coarser levels, which makes it
# not comparable to a language model's perplexity. The joint over all levels is: the coarse codes are deterministic functions of the
# bytes, so the sum over levels of (CE x slots) is an upper bound on the code length of the bytes under this model. Dividing by the 256
# bytes gives bits per byte, the unit byte language models report. For scale, a unigram byte model sits at about 5 bits per byte and
# a strong byte LM at about 1 to 1.5.

# %%
@torch.no_grad()
def _bits_per_byte():
    net.eval()
    tot = {}
    for mb in VAL:
        r = mb["rtype"]
        if r not in ("text0", "text"):
            continue
        ce = _ce_levels(mb, mb["cond"], mb["codes"])                       # nats per code at each level (from the 12b cell)
        n = len(mb["src"])
        nats = sum(c * l for c, l in zip(ce, LAYOUTS[r].lens))             # nats per row over all levels
        a = tot.setdefault(r, [0.0, 0.0, 0]); a[0] += nats * n; a[1] += ce[-1] * LAYOUTS[r].lens[-1] * n; a[2] += n
    for r, (s, s_last, n) in tot.items():
        print(f"  {r:6s} {s / n / SPEC.text / math.log(2):.3f} bits/byte over all levels (upper bound) | finest level alone {s_last / n / SPEC.text / math.log(2):.3f} | {n} held-out rows")
    u = UNI.get(("text", SPEC.K - 1))
    if u is not None and float(u.sum()) > 0:
        p = (u / u.sum())[1:]; p = p[p > 0]
        print(f"  unigram byte entropy of the training stream: {float(-(p * p.log2()).sum()):.3f} bits/byte")

_bits_per_byte()
