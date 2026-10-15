# Revision history

> This guide has two editions.
> The living web and PDF edition updates as findings are added or corrected; the arXiv snapshot carries a fixed version number.
> Each edition is numbered and dated; v1 is the original release and the arXiv edition is the stable point to cite.

The web edition and the PDF rebuild whenever a finding is added, corrected, or hardened, and each edition is numbered and dated on the title page, beginning with v1.
The arXiv edition, arXiv:2606.22283, is a snapshot taken at milestones, so its version number may lag the current edition after v1.
The revisions below are listed newest first; the ones promoted to an arXiv version are marked.

Each entry separates additions and corrections to the text from changes in the evidentiary status of a claim, since a claim moving from predicted or decompile-derived to measured is the change a returning reader most needs to see.

## v3 (October 14, 2026)

- Corrected the fp16 saturation figures in chapter 3: on the elementwise-multiply path the true fp16 limit is 65504, not 65536, and the multiply-accumulate ceiling at 32768 sits just above half the fp16 maximum rather than exactly at half of it.
- Corrected the convolution cost step in chapter 20, where 192 to 256 output channels is a 1.33-fold increase in arithmetic and had been given as 1.8-fold.
- Corrected the energy comparisons in chapter 13, which gave the ResNet-18 and twelve-layer-encoder results as multiples of energy rather than of energy efficiency and so inverted their sense against the same two figures in chapter 11.
- Corrected the double-multiply eligibility rule in chapter 36: the mode applies when the activation-float and kernel-float bits agree, which is when their exclusive-or is zero.
- Corrected the conv weight-gradient saturation entry in chapter 15, where 65504 divided by 16 is exact.
- Corrected the op-creation call order in section 12.6, which now passes the out-parameter first as the runtime header declares.
- Appendix C no longer directs readers to an unpublished research corpus. The tables too large to print are now described as held in the collected data and not reproduced here.
- Editorial pass over the whole guide: varied how tables, listings and figures are introduced, removed the chapter-summary restatements that the following paragraph repeated, cut prose that re-read the rows of an adjacent table, thinned repeated statements of a single measurement, and reduced the contrastive construction that had become the default emphasis throughout. Comma splices, dropped articles, inconsistent numeral style and one British spelling were repaired along the way.
- Rewrote the abstract and the opening of the introduction.
- Added a statement on the use of AI tools in preparing the guide.
- Evidence: no claim changed evidentiary status in this edition. Every measured, decompile-derived and predicted mark stands as it did in v2.

## v2 (June 24, 2026)

- Added section 35.11, a cross-generation comparison of the task-descriptor program. One convolution and one matmul are cross-compiled for the five base M-series targets, M1 through M5, from a single compiler build, and the codegen-revision stamp, the convolution op-config word, and the relocation-record stride are read per generation.
- Added a cross-source corroboration of the chapter 20 datapath geometry against Handley's patent-based architectural analysis, with the work added to the references.
- Evidence: the per-generation versioning of the task descriptor, given in chapter 23 as decompile-derived from the descriptor accessors, is now also confirmed against the emitted byte stream across M1 through M5.

## v1 (June 21, 2026)

- Initial public release, arXiv:2606.22283 v1.
