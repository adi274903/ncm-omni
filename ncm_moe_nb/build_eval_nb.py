#!/usr/bin/env python
"""Assemble the evaluation-only notebook: the setup / codec / model cells of the training notebook, then a checkpoint load and only
evaluation cells (no training, no benchmark, no correctness suite).

    python build_eval_nb.py --out NCM_MoE_Eval.ipynb

The cells come from src/ (shared with the training notebook, so the model and codecs are built by the same code) and eval_src/ (new).
"""
import argparse, os, sys
import nbformat
from run_src import cells_of, HERE
from build_nb import md_text

# order matters: E1 loads the weights after the model and evaluate() exist; E2 rebuilds the baseline before the learned report;
# generate() (13_generate) must exist before E3.
PLAN = [
    ("eval_src", "E0_env"),
    ("src", "00_dotenv"), ("src", "01_setup"), ("eval_src", "E0b_guard"), ("src", "02_hardware"), ("src", "03_spec"),
    ("src", "06_codecs"), ("src", "07_data"), ("src", "07a_tokenizer"), ("src", "06b_feat_codecs"), ("src", "07b_shards"),
    ("src", "07c_stage"), ("src", "07d_codec_checks"),
    ("src", "04_kernels"), ("src", "05_model"), ("src", "08_setup"), ("src", "09_budget"), ("src", "11_train"),
    ("eval_src", "E1_load_ckpt"), ("eval_src", "E2_unigram"),
    ("src", "12_verdict"), ("src", "12b_learned"),
    ("eval_src", "E5_bpb"), ("eval_src", "E4_source_gap"),
    ("src", "13_generate"), ("src", "13b_gencheck"),
    ("eval_src", "E3_audio_dac"),
]


def build():
    nb = nbformat.v4.new_notebook()
    for where, stem in PLAN:
        path = os.path.join(HERE, where, stem + ".py")
        for kind, text in cells_of(path):
            nb.cells.append(nbformat.v4.new_markdown_cell(md_text(text)) if kind == "md" else nbformat.v4.new_code_cell(text.strip("\n")))
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
                   "accelerator": "GPU"}
    nbformat.validate(nb)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "NCM_MoE_Eval.ipynb"))
    a = ap.parse_args()
    nb = build()
    nbformat.write(nb, a.out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {a.out}: {len(nb.cells)} cells ({n_code} code), {sum(len(c.source.splitlines()) for c in nb.cells):,} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
