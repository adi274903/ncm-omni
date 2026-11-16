#!/usr/bin/env python
"""Assemble the v2 training notebook: the same cells as NCM_MoE_Omni_Shards.ipynb (src/), preceded by one cell that sets the v2 knobs
(IMAGE_RES=v2, IMAGE_CODEC=vq: 16,384-bit images from a learned VQ with two codebooks at the 32x32 grid).

    python build_v2_nb.py --out ~/Downloads/NCM_MoE_V2_16k.ipynb
"""
import argparse, os, sys
import nbformat
from run_src import all_cells, cells_of, HERE
from build_nb import md_text


def build():
    nb = nbformat.v4.new_notebook()
    cells = [(k, t) for k, t in cells_of(os.path.join(HERE, "v2_src", "V0_env.py"))] + [(k, t) for _, k, t in all_cells()]
    for kind, text in cells:
        nb.cells.append(nbformat.v4.new_markdown_cell(md_text(text)) if kind == "md" else nbformat.v4.new_code_cell(text.strip("\n")))
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
                   "accelerator": "GPU"}
    nbformat.validate(nb)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "NCM_MoE_V2_16k.ipynb"))
    a = ap.parse_args()
    nb = build()
    nbformat.write(nb, a.out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {a.out}: {len(nb.cells)} cells ({n_code} code), {sum(len(c.source.splitlines()) for c in nb.cells):,} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
