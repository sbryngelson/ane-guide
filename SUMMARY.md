# Summary

## Front half — Using the ANE

### Part I — The Machine

- [01. What the ANE is](part-1-machine/01-what-the-ane-is.md)
- [02. Execution model](part-1-machine/02-execution-model.md)
- [03. Numerics](part-1-machine/03-numerics.md)
- [04. Capability surface](part-1-machine/04-capability-surface.md)

### Part II — Reaching the ANE

- [05. Software stack](part-2-reaching/05-software-stack.md)
- [06. Dispatching without Core ML](part-2-reaching/06-dispatch-without-coreml.md)
- [07. Weights and compression](part-2-reaching/07-weights-and-compression.md)
- [08. Entitlement boundary](part-2-reaching/08-entitlement-boundary.md)

### Part III — Performance and Fit

- [09. Roofline](part-3-performance/09-roofline.md)
- [10. Power and efficiency](part-3-performance/10-power-efficiency.md)
- [11. ANE, GPU, and CPU](part-3-performance/11-ane-vs-gpu-vs-cpu.md)
- [12. Across the chip family](part-3-performance/12-cross-chip.md)

### Part IV — Workloads

- [13. Vision, convolution, and encoders](part-4-workloads/13-vision-conv-encoders.md)
- [14. LLM case study](part-4-workloads/14-llm-case-study.md)
- [15. Training on the engine](part-4-workloads/15-training.md)
- [16. Numerical and scientific computing](part-4-workloads/16-numerical-scientific.md)

### Part V — Practice

- [17. Model-design rules](part-5-practice/17-model-design-rules.md)
- [18. Optimization and the cost model](part-5-practice/18-optimization-cost-model.md)
- [19. Pitfalls and limits](part-5-practice/19-pitfalls-and-limits.md)

## Back half — Architecture and Internals

- [Interlude. Below the API: how the engine works](part-5-practice/handoff.md)

### Part VI — The Silicon

- [20. Datapath and MAC geometry](part-6-silicon/20-datapath-mac.md)
- [21. Memory hierarchy](part-6-silicon/21-memory-hierarchy.md)

### Part VII — The Toolchain and Encoding

- [22. Compiler](part-7-toolchain/22-compiler.md)
- [23. Program and container format](part-7-toolchain/23-program-format.md)
- [24. HAL and capability gates](part-7-toolchain/24-hal-gates.md)
- [25. Compression internals](part-7-toolchain/25-compression-internals.md)
- [26. Hidden layers and direct netplist authoring](part-7-toolchain/26-hidden-layers.md)

### Part VIII — System Internals

- [27. Kernel driver and IOKit ABI](part-8-system-internals/27-kernel-driver.md)
- [28. Address translation and the DART](part-8-system-internals/28-dart-iommu.md)
- [29. Firmware](part-8-system-internals/29-firmware.md)
- [30. Host-to-firmware command protocol](part-8-system-internals/30-csne-protocol.md)
- [31. Power and thermal](part-8-system-internals/31-power-thermal.md)
- [32. Security and isolation](part-8-system-internals/32-security.md)
- [33. Telemetry and hardware counters](part-8-system-internals/33-telemetry.md)

### Part IX — Cross-Silicon Reference

- [34. Cross-silicon targets](part-9-cross-silicon/34-the-28-targets.md)
- [35. Per-family code generation](part-9-cross-silicon/35-per-family-codegen.md)
- [36. Predicted upper tier](part-9-cross-silicon/36-upper-tier.md)

## Back matter

- [Methodology](back-matter/methodology.md)
- [Open questions](back-matter/open-questions.md)
- [Statements](back-matter/statements.md)

## Appendices

- [Appendix A. Operation-by-device matrix](appendices/a-op-device-matrix.md)
- [Appendix B. Hidden-layer catalog](appendices/b-hidden-layer-catalog.md)
- [Appendix C. Decoded reference tables](appendices/c-decoded-tables.md)
- [Appendix D. Glossary](appendices/d-glossary.md)
- [Appendix E. Provenance](appendices/f-provenance.md)

## References

- [References](references.md)
