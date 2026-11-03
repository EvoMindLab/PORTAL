# PORTAL user documentation

`UserManual.pdf` is the distributable Studio guide. Its source is
`UserManual.tex`, with the small bibliography `manual_references.bib` and
screenshots in `../Figs_manual/`. No manuscript sources or AUX files are needed.

From this directory, build with an installed LaTeX distribution:

```bash
pdflatex -interaction=nonstopmode -halt-on-error UserManual.tex
bibtex UserManual
pdflatex -interaction=nonstopmode -halt-on-error UserManual.tex
pdflatex -interaction=nonstopmode -halt-on-error UserManual.tex
```

Only the guide, its source, this bibliography, and this README are part of the
public `Documentation/` directory. Research manuscripts and experimental
artifacts are not included. The command-line runner references live under
`Recipe_Selection/suite_pipeline/` and `Recipe_Selection/augmentation_pipeline/`.
