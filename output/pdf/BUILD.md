# Inference chip product roadmap

- `inference_chip_product_roadmap.pdf`: compiled reading copy.
- `inference_chip_product_roadmap.tex`: editable, self-contained LaTeX source.

Build from this directory using a TeX installation (for example TeX Live/MacTeX):

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error inference_chip_product_roadmap.tex
```

The document uses standard TeX packages listed in the preamble, including newtx,
microtype, tabularx, tcolorbox, TikZ and hyperref. No model weights or inference
program are executed to build the PDF. If latexmk is not available, run pdflatex
at least twice to resolve the contents, cross-references and total page count.

This is a dated engineering roadmap and repository snapshot, not an assertion
that the full model or chip is complete. Implementation gates, numerical caveats,
physical design and manufacturing requirements are detailed in the document.
