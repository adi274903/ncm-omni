#!/usr/bin/env python
"""Assemble NCM_MoE_Omni_Shards.ipynb from src/*.py (percent-format cells).

    python build_nb.py            # write the notebook
    python build_nb.py --check    # exit 1 if the notebook is stale or not valid nbformat
The notebook is self-contained: it imports nothing from this repo.
"""
import argparse, json, os, sys
import nbformat
from run_src import all_cells, HERE

OUT = os.path.join(HERE, "NCM_MoE_Omni_Shards.ipynb")


def md_text(t):
    lines = []
    for l in t.strip("\n").split("\n"):
        lines.append(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l))
    return "\n".join(lines).strip("\n")


def build():
    nb = nbformat.v4.new_notebook()
    for _, kind, text in all_cells():
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(md_text(text)))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(text.strip("\n")))
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
                   "accelerator": "GPU"}
    nbformat.validate(nb)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    nb = build()
    if a.check:
        old = nbformat.read(a.out, as_version=4)
        same = [c.source for c in old.cells] == [c.source for c in nb.cells]
        print("OK: notebook matches src/" if same else "STALE: rebuild with python build_nb.py")
        return 0 if same else 1
    nbformat.write(nb, a.out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {a.out}: {len(nb.cells)} cells ({n_code} code), {sum(len(c.source.splitlines()) for c in nb.cells):,} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
