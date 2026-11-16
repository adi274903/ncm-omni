#!/usr/bin/env python
"""Run the notebook source as a plain script (same cells, one namespace). Usage:
    python run_src.py [--upto NN] [--extra file.py]   # env NCM_* overrides apply
"""
import argparse, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cells_of(path):
    """-> [(kind, text)] from a percent-format file."""
    out, kind, buf = [], None, []
    for line in open(path).read().split("\n"):
        m = re.match(r"# %%(\s*\[markdown\])?\s*$", line)
        if m:
            if kind is not None:
                out.append((kind, "\n".join(buf)))
            kind, buf = ("md" if m.group(1) else "code"), []
        else:
            buf.append(line)
    if kind is not None:
        out.append((kind, "\n".join(buf)))
    return out


# Cell order. Everything the encode stage needs (setup, spec, codecs, data, shards) comes first; the kernel, model and
# build sections only matter for training and are skipped when SHARD_MODE="encode". Files not listed run last, by name.
ORDER = ["00_dotenv", "01_setup", "02_hardware", "03_spec", "06_codecs", "07_data", "07a_tokenizer", "06b_feat_codecs", "07b_shards", "07c_stage",
         "07d_codec_checks", "08b_encode", "04_kernels", "05_model", "08_setup", "09_budget", "10_correct", "10b_profile", "11_train", "12_verdict",
         "12b_learned", "13_generate", "13b_gencheck", "14_storage"]


def all_cells():
    cells = []
    files = glob.glob(os.path.join(HERE, "src", "*.py"))
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    for f in sorted(files, key=lambda f: (ORDER.index(stem(f)) if stem(f) in ORDER else len(ORDER), stem(f))):
        cells += [(f, k, t) for k, t in cells_of(f)]
    return cells


def plain(text):
    """IPython magics -> plain Python, so the same cells also run as a script (`%env K=V` -> os.environ, other magics dropped)."""
    out = []
    for l in text.split("\n"):
        s, ind = l.strip(), l[:len(l) - len(l.lstrip())]
        if s.startswith("%env ") and "=" in s:
            k, _, v = s[5:].partition("=")
            out.append(f"{ind}os.environ[{k.strip()!r}] = {v.strip()!r}")
        elif s.startswith(("%", "!")):
            out.append(f"{ind}pass")
        else:
            out.append(l)
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", default="99")
    ap.add_argument("--extra", default=None, help="python file exec'd in the namespace afterwards")
    a = ap.parse_args()
    ns = {"__name__": "__main__", "os": os}
    for f, k, t in all_cells():
        if k != "code" or os.path.basename(f)[:2] > a.upto:
            continue
        exec(compile(plain(t), f"{os.path.basename(f)}", "exec"), ns)
    if a.extra:
        exec(compile(open(a.extra).read(), a.extra, "exec"), ns)
