## Apple Neural Engine: Architecture, Programming, and Performance

[![Documentation](https://readthedocs.org/projects/ane-guide/badge/?version=latest)](https://ane-guide.readthedocs.io/en/latest/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22283-b31b1b.svg)](https://arxiv.org/abs/2606.22283)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This guide documents the Apple Neural Engine (ANE), the fixed-function matrix accelerator built into Apple silicon from the A11 and M1 generations onward.
It reports what Apple's public surface does not: the datapath and roofline, the dispatch route below Core ML, the compiler and program format, and the kernel driver, firmware, and command protocol, each from measurement and decompilation.
Every substantive claim is marked measured, decompile-derived, or predicted.

### Reading it

The web edition is at <https://ane-guide.readthedocs.io>, and the PDF is on arXiv ([arXiv:2606.22283](https://arxiv.org/abs/2606.22283)).
Each chapter opens with a short summary, then the mechanism, then a worked example (front half) or the reference tables, structs, and register maps (back half).
Parts I through V deploy and tune a model; Parts VI through IX and the appendices document the engine itself: the datapath, memory hierarchy, compiler, program format, kernel driver, firmware, command protocol, and per-chip target tables.
The full table of contents is in [SUMMARY.md](SUMMARY.md).

### Status and stability

The direct route this guide describes is reachable from ordinary user space, but reachable does not mean supported.
The private runtime, compiler, and symbols named here are not a public Apple interface.

- Reachable: the operations and the dispatch path compile and run today on the measured silicon.
- Unsupported and private: none of this is a documented Apple interface, and Apple makes no compatibility promise.
- Version-fragile: the private symbols and program formats can change or break on any operating-system update.
- Not App Store safe: an application that links these private interfaces may be rejected from App Store distribution.
- Not redistributable: the private frameworks belong to the operating system.

The supported path for shipping software remains Core ML.
The direct route is for measurement, research, and on-device work on a fixed operating-system version.

### How to cite

This guide is archived on arXiv as [arXiv:2606.22283](https://arxiv.org/abs/2606.22283).

```bibtex
@misc{bryngelson2026ane,
  author        = {Bryngelson, Spencer H.},
  title         = {Apple Neural Engine: Architecture, Programming, and Performance},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {2606.22283},
  doi           = {10.48550/arXiv.2606.22283},
  url           = {https://arxiv.org/abs/2606.22283},
}
```
