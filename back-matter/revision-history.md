# Revision history

> This guide has two editions: a living web and PDF edition that updates as findings are added or corrected, and a periodic arXiv snapshot that carries a fixed version number.
> Each web and PDF build is dated on the title page; the arXiv edition is the stable point to cite.

The guide is maintained as a living document.
The web edition at the project site and the PDF rebuild whenever a finding is added, corrected, or hardened, and each build is stamped with its date on the title page.
The arXiv edition, arXiv:2606.22283, is a periodic snapshot taken at milestones rather than on every change; it carries a fixed version number and is the stable point to cite.
The revisions below are listed newest first and dated; the ones promoted to an arXiv version are marked.

Each entry separates additions and corrections to the text from changes in the evidentiary status of a claim, since a claim moving from predicted or decompile-derived to measured is the change a returning reader most needs to see.

## 2026-06-24

- Added section 35.11, a cross-generation comparison of the task-descriptor program. One convolution and one matmul are cross-compiled for the five base M-series targets, M1 through M5, from a single compiler build, and the codegen-revision stamp, the convolution op-config word, and the relocation-record stride are read per generation.
- Added a cross-source corroboration of the chapter 20 datapath geometry against Handley's patent-based architectural analysis, with the work added to the references.
- Evidence: the per-generation versioning of the task descriptor, given in chapter 23 as decompile-derived from the descriptor accessors, is now also confirmed against the emitted byte stream across M1 through M5.

## 2026-06-21 (arXiv v1)

- Initial public release, arXiv:2606.22283.
