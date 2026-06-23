# Appendix E. Provenance

This appendix records the evidentiary basis of every substantive claim in the guide, Part by Part and chapter by chapter.
The method is the one given in the Methodology chapter: the engine was reached directly below Core ML, the stack was read by static decompilation of the runtime, compiler, kernel driver, and firmware, and values were taken by live read-only instrumentation and by compile-and-run probing.
Two silicon points were measured directly, M1/H13 (Apple M1) as the primary host and M5/H17s (Apple M5) as the second, with an A14-class part (Apple M2, H14) as a middle point where a claim required one.
Each claim below is marked one of three ways: measured on a named generation, decompile-derived from a named binary, or predicted from the per-chip tables and not yet confirmed on silicon.

## Part I. The Machine {.unnumbered}

Part I was measured on M1/H13 (Apple M1) unless a chapter notes otherwise.

### Chapter 1. What the ANE is {.unnumbered}

Apple documents the engine only as the `MLComputeUnits` compute-unit selector at developer.apple.com/documentation/coreml/mlcomputeunits, a placement hint with no direct device API and no way to confirm which unit ran a segment; this chapter extends that account by reaching the engine directly below the selector.
The roofline figures and the convolution advantage over the GPU (3.8x faster, 9x more energy efficient) are M1/H13 measured.
The fp16-product and wide-accumulator result is decompile-derived from the firmware and the compiler and confirmed by the M1/H13 cancellation probe.
The Core ML placement planner and the direct Espresso route are decompile-derived.

### Chapter 2. Execution model {.unnumbered}

Apple documents only the load-and-predict surface (`MLModel`, `prediction(from:)`); the autonomous-coprocessor model in this chapter, with its command mailbox, walked segment graph, and resident state across dispatches, has no public counterpart and is reported as reverse-engineered.
The compile-once, dispatch-many split and the disk-cached program are decompile-derived and confirmed by M1/H13 dispatch tracing.
The mailbox-and-doorbell command channel, the autonomous controller, operand mapping through the address-translation unit, and the walked segment graph with its static-control-flow consequence are decompile-derived from the firmware, kernel-driver, and program-format work.
The resident-state mechanism through output-to-input buffer aliasing is M1/H13 measured, observed to persist and update an accumulator and a key-value cache across successive dispatches with no host resubmission.

### Chapter 3. Numerics {.unnumbered}

The fp16 datapath and the type limits are decompile-derived from the firmware and the compiler, with the activation-table coefficients read out of the compiler binary constant section.
The wide accumulator and its radix-4 first stage, activation-table behavior and the twenty-three-op accuracy sweep, NaN coercion and the gelu and swish origin biases, round-half-to-even output grid, MAC saturation at exactly $2^{15}$, denormal handling, softmax max-subtraction, and bit-deterministic output for a fixed graph and input are all M1/H13 measured.
The slice saturation threshold at 4094 is M1/H13 measured and reproduced on A14/H14 (Apple M2), where 4094 stays finite and 4096 overflows; the clean route arrives at A15 and later.
The cross-generation accumulator behavior holds family-wide, the M5/H17s difference being a one-unit-in-the-last-place scheduling and tiling reorder rather than an accumulator-width change.
Denormal preservation inside the M5/H17s accumulator, where the M1 flushes to zero inside the multiply-accumulate, is M5/H17s measured and generational.
Apple's conversion tooling documents low-precision compression at apple.github.io/coremltools, and the int8 relation here agrees with its affine form $w = s\,(q - z)$; the wide-accumulator and per-product cancellation results have no public counterpart and are reported as reverse-engineered.

### Chapter 4. Capability surface {.unnumbered}

The native classes, type limits, compile-legal envelope, and M1 limits are M1/H13 measured by operation-conformance runs, each native operation compiled and run and each unsupported one rejected on device.
The attested-is-not-reachable rule is M1/H13 measured: three-dimensional convolution fails backend lowering on every device mask despite its capability byte, and the top-k, sort, and dynamic-slice validators are callable but code-generation-rejected on the M1.
The cumsum result is M1/H13 measured through the curated runtime path, correcting the earlier no-path status.
The unsupported-on-every-family set and the per-family unlock points for the texture-engine operations, sin and cos, and the rank and sort bridge are decompile-derived from the operation floors and validators and confirmed on the M1 only at its boundary; the chip that first runs each is predicted from the floor table.
Apple documents the convertible operation set at apple.github.io/coremltools and developer.apple.com; this chapter extends and partly corrects it, reporting what compiles and runs on the direct path rather than what the public converter accepts, since some accepted operations, such as three-dimensional convolution, do not lower to the engine.

## Part II. Reaching the ANE {.unnumbered}

Part II was measured on M1/H13 (Apple M1) unless a chapter notes otherwise, with chapter 7 also measured on M2/H14 (Apple M2 Pro) and M5/H17s (Apple M5).

### Chapter 5. Software stack {.unnumbered}

Apple documents the compute-unit selector, model loading, and the placement read-out (`MLComputePlan.load`, `deviceUsage(for:)`, `estimatedCost(of:)`) at developer.apple.com/documentation/coreml, and that read-out agrees with the segmenter decision reported here; the layered runtime beneath it and its internal cost graph are not documented and are reported as reverse-engineered.
The layering and the runtime's ownership of compile, library, stream, and descriptors; the shortest-path placement segmenter with its per-operation-by-backend cost graph, learned decision trees, and launch and transfer penalties; the broker model with its single privileged device gate, content-hashed program cache, and time-shared request queue; the 292-export runtime surface and its options dictionary; and the per-inference submit through `IOConnectCallAsyncMethod` selector 2 with the daemon's lifecycle selectors 3 through 6 are decompile-derived from the framework, runtime, and daemon binaries.
The reachability of the runtime below the framework, with no placement planner and no entitlement for accepted operations, is decompile-derived and confirmed by M1/H13 dispatch tracing.

### Chapter 6. Dispatching without Core ML {.unnumbered}

Apple documents only the indirect `MLModel` and `prediction(from:)` route; the direct compile, load, bind, and dispatch route here is reported as reverse-engineered and extends that account with a planner-free path that targets the engine on purpose.
The five-step workflow and the absence of a placement planner are decompile-derived from the runtime and the format pipeline.
The absence of an entitlement requirement for accepted operations is decompile-derived and confirmed M1/H13 measured from ordinary user space, and the single-submission multi-step drive with its performance-neutral result and the resident-state buffer aliasing it rests on are M1/H13 measured.

### Chapter 7. Weights and compression {.unnumbered}

This chapter measures across three endpoints: M1/H13 (Apple M1), A14 on M2/H14 (Apple M2 Pro), and M5/H17s (Apple M5).
The int4 and sparse streaming speedups and byte ratios, the int8 matmul latency and weight-byte halving, and the M1 folding of the int8 and blockwise forms are M1/H13 measured.
The A14 int8 and sparse stream and the A14 blockwise fold are M2/H14 measured, and the streaming of all four forms is M5/H17s measured.
The weight-reconstruction codecs and the per-format hardware-abstraction-layer streaming gates are decompile-derived from the ANE compiler; the A15 floor at which the blockwise form first streams is predicted from the gate pattern and not yet silicon-confirmed.
Apple's conversion tools document palettization, quantization, and pruning at apple.github.io/coremltools as a model-size feature; this finding extends that account, showing the same forms stream on the unentitled engine for a bandwidth gain.

### Chapter 8. Entitlement boundary {.unnumbered}

The four gated features and the layer each gate is at are M1/H13 measured: three-dimensional convolution fails backend lowering on every device mask, a native-state program fails the compile with the counter-and-event engine stubbed in the M1 descriptor, a bf16 input or output fails with an unsupported-dtype rejection and is absent from the eleven-code program-I/O enumeration, and a symbolic dimension parses but fails to lower.
The framework cast of bf16 to fp16 and the bucketed fixed-shape handling of flexible shapes are decompile-derived from the framework and system bundles, and the load-time signature boundary is decompile-derived from the kernel driver's corecrypto signature and trustcache vnode-trust checks, with the load rejection code `0xe00002e2` observed on the M1.
The arrival of native resident state on a later generation is predicted from the per-family hardware descriptor and not measured on that silicon.
Apple documents these capabilities (flexible shapes, the `MLState` type) at the conversion and runtime layer; the finding shows they are not reachable on the unentitled direct path, correcting the impression that a documented capability runs on the engine directly.

## Part III. Performance and Fit {.unnumbered}

Part III was measured on M1/H13 (Apple M1; M1 Max), with a second pass on M5/H17s (Apple M5) and an A14-class middle point on M2/H14 where a cross-device or cross-generation claim required it.
Apple publishes no roofline, no per-workload power or efficiency figures, and no cross-processor comparison for the engine, so the figures across this Part are reported as measured rather than against a documented account.

### Chapter 9. Roofline {.unnumbered}

The M1 compute and bandwidth ceilings, the 141 FLOP-per-byte ridge point, 2 MB working-set threshold, 0.23 ms dispatch floor, and conv-throughput scaling are M1/H13 measured from the memory-controller and energy counters and end-to-end timing.
The saturating-method figures (the large-matmul compute ceiling, effective peak, wall-clock weight-stream bandwidth matching the compiler's internal 50 GB/s constant, int8 rate, and full-call floor) are M1/H13 measured live in a read-only dtrace session, and are presented alongside the counter-based and slope-based figures with the methodology difference noted.
The cross-device ridge points, the standalone and weight-stream bandwidths, and fused-block rate are M5/H17s measured.
The fusion-and-floor model is decompile-derived from the analytic cost model and fit to the measured M1 convs within plus or minus 17 percent; cross-chip scaling of the ceilings is predicted from that model, not asserted as a measured M1 fact.

### Chapter 10. Power and efficiency {.unnumbered}

The convolution-stack efficiency figures, the absolute-power comparison on the 4096 matrix multiply, the 2-to-14-times efficiency range, and sustained-load behavior are M1/H13 measured, with package power from hardware instrumentation.
The power-utilization model (the zero idle rail, dispatch floor, fp16 and int8 compute-bound draw, regime-dependent points, and operations-per-watt optimum near 0.37 pJ per FLOP) is M1 Max measured with the root power sampler, which exposes only a power reading and a binary on-or-off state, with no frequency or voltage telemetry.
The A14-class middle generation is measured on that silicon, and the M5 efficiency figures are M5/H17s measured.

### Chapter 11. ANE, GPU, and CPU {.unnumbered}

The per-class speed and energy verdicts are M1/H13 and M5/H17s measured by a single sixteen-class harness recording minimum latency, idle-subtracted total-package power, and fp16 relative error per class.
The saturation peaks, the large-N matrix-multiply falloff, and serving crossovers are M5/H17s (Apple M5 Pro) measured; the per-eval overhead floor and the M1 power gap are M1/H13 measured.

### Chapter 12. Across the chip family {.unnumbered}

The naming rule $M(n) \rightarrow H(n+12)$ is decompile-derived from the per-family device tables and confirmed on the measured parts: M1/H13 by the live h13g architecture string, M2/H14g on the live A14 host, and M5/H17s by the resolved target.
The family-wide operation limits and the core-and-clock scaling are decompile-derived from the operation floors and the per-target scalar tables.
The M5 throughput and working-set threshold, the ten-of-ten cross-silicon prediction pass, the M1-versus-M5 training and inference parity (0.9080 against 0.9070, deterministic across runs), and the one-unit-in-the-last-place fp16 divergence bound are M5/H17s and M1/H13 measured; the M2 campaign measured training accuracy, the four fp16 axes, peak throughput, compression, and the per-op max-dim caps on A14 silicon.
The A15 and A16 generations and their M3 and M4 counterparts are decompile-derived from the device tables and not individually measured, the A15/M3 rail being the one generation that remains unmeasured.
Apple's product specifications publish a marketing core count per chip, for example a 16-core engine, a different quantity from the four physical compute sets measured on the M1; the H-architecture naming, the mapping, and per-family gates are reported as reverse-engineered.
The M5 single-program matmul peak of about 9.5 fp16 TFLOP/s and the weight-stream bandwidth of about 145 GB/s over two DRAM read channels are M5/H17s measured.

## Part IV. Workloads {.unnumbered}

Part IV was measured on M1/H13 (Apple M1 Max) and M5/H17s (Apple M5 Pro), with an A14-class point on M2/H14 where noted, each figure marked for the generation it was taken on.

### Chapter 13. Vision, convolution, and encoders {.unnumbered}

The convolution speed and efficiency, the convolution-stack and roofline figures, the power gap, and per-eval floor are M1/H13 measured; the M5 convolution-stack, ResNet-18, twelve-layer-encoder, and single-sentence-encoder ratios and the serving crossovers are M5/H17s measured.
The convolution lowering, Winograd gate, 2 MB working-set constant, per-output-channel fold, and texture-engine operation set with its single A14 family gate are decompile-derived from the ANE compiler and the per-chip parameter table.
The Q.4 crop-scale saturation threshold, where $4094 \times 16 = 65504$ reaches the fp16 ceiling, is decompile-derived and confirmed by the fp16 range probe on M1/H13 and on A14 (M2); the clean route arrives on A15 and later.
Apple documents vision and image models on the engine only through the compute-unit selector (developer.apple.com/documentation/coreml, developer.apple.com/documentation/vision), with no direct datapath API or cross-processor figures; this chapter extends that account with the measured economics against the GPU and names the texture-engine preprocessing path the public surface does not expose.

### Chapter 14. LLM case study {.unnumbered}

The per-eval dispatch floor, int8-hybrid result, and resident-cache step are M1/H13 measured, the last by a two-proof resident-buffer probe; the batched-decode comparison, the per-projection placement table and position rule, and the speculative-decoding and batched-prefill serving controls are M5/H17s measured.
The dispatch and resident-state machinery of the direct path is decompile-derived from the runtime, including the output-to-input buffer aliasing that holds the cache resident, the in-flight cap of 127, and per-process loaded-program cap near 128 (the next load returns `GetANEFModel: must re-compile`).
Apple does not publish engine-versus-GPU decode measurements, so the per-batch decode verdict is reported as measured.

### Chapter 15. Training on the engine {.unnumbered}

The gradient audit, differentiable-vocabulary correctness to a cosine of $1.0000$, conv weight-gradient saturation threshold, M1 training accuracy and loss-scale curves, and two-generation parity (0.9080 on M1 against 0.9070 on M5 for the identical seeded network) are M1/H13 and M5/H17s measured.
The width-axis slice saturation, exact at loss scale 384 and first overflowing at 512 above $65504 / 16 \approx 4094$, was reproduced on A14 (M2), locating the clean route on A15 and later.
The absence of an engine-native backward operation is decompile-derived from the shared compiler, which has gradient operations only in the graphics-processor dialect.
Apple documents on-device model update at developer.apple.com/documentation/coreml, a limited fine-tuning surface whose backward pass runs off the engine; this chapter extends that account with a full forward, backward, and optimizer loop running as engine graph operations, optimizer state resident across steps.

### Chapter 16. Numerical and scientific computing {.unnumbered}

The iterative-solver envelope, the size bounds on the unrolled factorizations, the full-spectral-decomposition relative errors, and wide-accumulator behavior are M1/H13 measured, with the fused five-point stencil margin measured on M5/H17s and the DFT-as-matmul throughput on the M2 generation at $N = 1024$.
The static-dataflow constraint and the absence of data-dependent control flow are decompile-derived from the operation set and the compiler.
The fp16-clean DFT bound near $N = 2048$ is predicted from the wide-accumulator reduction and the fp16 rounding of the matrix entries, an edge of the representable range consistent with the accumulator measurements rather than a single measured cutoff.
Apple documents dense linear algebra and signal processing through the accelerate framework at developer.apple.com/documentation/accelerate, all targeting the CPU rather than the engine; this chapter maps which of those kernels fit the engine and which are architecture-limited, a mapping the public documentation does not provide.

## Part V. Practice {.unnumbered}

Part V was measured on M1/H13 (Apple M1), with M5/H17s (Apple M5) as the cross-chip reference where a second generation is needed.

### Chapter 17. Model-design rules {.unnumbered}

The tensor-dimension boundaries, the convolution kernel and stride limits, the arg-min and arg-max 2048 cap, and pooling-window behavior are M1/H13 measured, swept until compile flipped from accept to reject.
The per-operation validators, the kernel-format and maximum-dimension fields, the divisibility checks, and working-set and kernel-memory budgets are decompile-derived and joined to those sweeps; the per-family unlock points for the texture-engine padding modes, the square-after-reduction mode, and sin and cos are decompile-derived from the per-chip support flags, confirmed on the M1 only at its boundary and predicted for the chip that first enables each.
On the public side, Apple documents the conversion-time constraints and supported converter configurations at apple.github.io/coremltools; this chapter reports the argument, shape, and mode limits the on-device validators enforce, which the public converter does not enumerate.

### Chapter 18. Optimization and the cost model {.unnumbered}

The convolution latency fit and the dispatch floor are M1/H13 measured (Apple M1; M1 Max), and the cross-chip bandwidth, floor, and peak re-fit are M5/H17s measured, with the core-scaled bandwidth confirmed by direct streaming at 57.5 GB/s against the M1's 10.4 GB/s.
The compiler's analytic cost functions (the cycles, roofline, and wall-time chain) are decompile-derived with the per-chip parameters walked live from the hardware-abstraction table; the cross-chip scaling of unmeasured targets is predicted from core-count and clock ratios.
The cost-model fidelity is M1/H13 measured: a median error near 31 percent with 11 of 68 shapes within plus or minus 17 percent, sound as an ordinal placement tool rather than an absolute-latency oracle, with attention unmodeled and the 9.0 GB/s bandwidth anchor held below the roughly 40 GB/s effective rate because it is jointly calibrated for the convolution fit.
There is no public counterpart: Apple does not publish the engine compiler's cost model or its per-chip parameters, so the model is reported here as decompile-derived and validated against M1/H13 and M5/H17s measured latency.

### Chapter 19. Pitfalls and limits {.unnumbered}

The slice-saturation threshold, dynamic-weight convolution batch boundary, compile-failure back-off, and four-character-code lowering limit are M1/H13 measured, and the A14 generation (M2) was measured to saturate on the same slice, locating the clean route on A15 and above and confirmed clean on the M5.
The per-target slice-lowering template and its DMA source-path and patch-width routines are decompile-derived, giving the times-16 fixed-point DMA format and the target-keyed slice route; the clean A15 route is decompile-derived from the per-family lowering and confirmed clean on the M5, not measured on A15 silicon directly.
These are failure modes of the private compiler and have no public counterpart: they are reported as measured on the M1 and reverse-engineered from the compiler binary.

## Part VI. The Silicon {.unnumbered}

Part VI was measured on M1/H13 (Apple M1; M1 Max, live architecture string `h13g`); the datapath geometry and memory hierarchy are decompile-derived from the per-chip hardware-abstraction table and the engine compiler, calibrated against the M1 anchors.

### Chapter 20. Datapath and MAC geometry {.unnumbered}

The core count of four, per-core throughput scaling of 1 to 4, power-rail step per core, radix-4 fan-in, wide accumulator, and output-channel-group pass-doubling threshold at 192 to 256 are M1/H13 measured.
The accumulator budget of eight and the lane widths of four and eight are decompile-derived and uniform across chips per the hardware-abstraction table, which was carved from the H13 firmware blob and calibrated at the core-count, cycle-divisor, and working-set offsets; the convolution-lowering and performance-model functions are decompile-derived from the ANE compiler (`GetNumOutputChannelsPerCycle`, `GetNumOutputChannelsPerAccumulator`, `ComputeMaxOcgSize`, `ZinMirNECoreAssignment`, `GetNumNeededNEsNextPow2`).
Apple publishes a marketing core count per chip, for example a 16-core M1, a different quantity from the decoded `num_nes` of four; the multiply-accumulate geometry has no public counterpart and is reported as reverse-engineered and measured.
The int8 compile flag is weight-only quantization that leaves the multiply-accumulate in fp16, neutral on compute-bound work and about 1.5 times faster only on weight-bandwidth-bound matmuls near a 4096-by-4096 weight, M5/H17s measured.

### Chapter 21. Memory hierarchy {.unnumbered}

The 2.28 to 2.34 MB threshold and the 64-byte throughput period are M1/H13 measured on the dispatch path, and the absence of a runtime replacement policy is an M1/H13 measured negative, with sequential and random re-reference order identical at every footprint.
The field values are decompile-derived: `0x1b8` (2 MB operand working set), `0x1c8` (64 banks), `0x1c0` (16-byte granule), `0x1f8` (2 MB stride ceiling), `0x1f0` (residency threshold, 0 on M1), and `0x288` (64 KB kernel store); the operand-size comparator, the bank function and conflict model, and inverted residency-buffer gate are decompile-derived from the named compiler routines, with the 2 MB boundary itself the operand-size comparator.
The A15-class, A16-class, and M5 values of field `0x1f0` are read from the same table by offset but predicted for those parts, the gate behavior confirmed on the M1 only.
The memory hierarchy, the bank function, and residency threshold have no public counterpart and are reported as reverse-engineered and measured.
The compiler name for field `0x1b8`, `MemCacheSize` (also `L2Size`) with its `fl2-size` override, is decompile-derived; on the M5 the working-set crossing is smooth in throughput and shows instead in DRAM energy per operation, bottoming near a 2 MB operand, M5/H17s measured.

## Part VII. The Toolchain and Encoding {.unnumbered}

Part VII was measured on M1/H13 (Apple M1; M1 Max, live string `h13g`), with chapter 25 extending to M2/H14 (Apple M2 Pro) and M5/H17s (Apple M5).
This Part is mostly decompile-derived from the engine compiler decompile and its constraint-string corpus, the per-chip hardware-abstraction table read by byte offset across 28 target entries, and the runtime serializers and task-descriptor setters; unless a chapter says otherwise, scalar offsets, capability-byte offsets, struct fields, and symbol names are decompile-derived.

### Chapter 22. Compiler {.unnumbered}

The four-phase pipeline, task-descriptor partition budget, allocation-type set, `anec.matmul` and `anec.convolution` lowerings, and fusion rules (the GOC fused unit, seven-slot epilogue, fusable epilogues and the two-live-input, concat, and attention-cut barriers) are decompile-derived from the engine compiler framework (the 9.509 build, a 4.1-million-line decompile) and the constraint-string corpus.
The validator export set, the per-layer reject strings, and the code-generation rejections for top-k, sort, dynamic-slice, and three-dimensional convolution are M1/H13 measured.
The compiler internals, the backend `anec.*` dialect, and the `_ANECValidate*` surface have no public counterpart and are reported as reverse-engineered; the public tools document only the frontend operation set and conversion passes.

### Chapter 23. Program and container format {.unnumbered}

The two-layer split, four-field root table, cast-inference-cast operation shape, seven register groups, 15-bit and 17-bit dimension widths, relocation-slot model, 44-byte sparse record, and the HWX on-disk layout (the `0xbeefface` Mach-O variant, the segment set, the `ZinAneTd` linked list, the per-lane weight tiles, and shape descriptors) are decompile-derived from the serializer and task-descriptor symbols and cross-confirmed against the vendor's task-descriptor symbol table.
The dispatch-descriptor schema was validated by a round-trip through the schema compiler (version 25.12.19), which regenerated an object-API header and a binary reflection schema without error.
The decoded identity-linear program with its segment map, port descriptors, register records, and weight bank are M1/H13 measured, parsed byte for byte from real on-disk files in the runtime caches, with the resolved tensor frame read from the post-compile status sidecar.
The format has no public counterpart and is reported as reverse-engineered; the full FlatBuffer schema is Appendix C.
The custom-bar bit-field relocation, which patches a resolved value into a named descriptor field by bit offset and width, and the range-checked live-in shape and stride parameters that let one program serve a range of input shapes, are decompile-derived from the program loader.

### Chapter 24. HAL and capability gates {.unnumbered}

The scalar offsets, the capability-byte offsets, and family floors are decompile-derived, with every per-target constructor invoked on this host (live string `h13g`) to read byte-exact values for all 28 targets, joined to the minimum-family operation trait and tier assignment read from the same binary.
The per-family unlock generations for the texture engine, sin and cos, and the dimension and format-count steps are decompile-derived from the per-target tables and confirmed on the M1 only at its boundary; the generation that first enables each is predicted from the floor table.
The attested-is-not-reachable rule is M1/H13 measured: three-dimensional convolution has its HAL kernel-depth attestation at offset `0x70` and fails backend lowering on every device mask, and the top-k, sort, and dynamic-slice validators are callable but code-generation-rejected.
The packed-bitfield struct measures `0x938` bytes; its non-flag residual is the cost-model coefficient block at `0x580` through `0x7f0` plus about two soft fp64 coefficients, and the offsets past `0x938`, read in an earlier round as an A12 operation-emulation catalog at `0xa30` through `0xe84`, are a read into zeroed memory beyond the struct and have no table; the capability-flag offsets are decompile-derived from the `ZinIrHalParameters` reader symbols.
The HAL table, the capability-byte region, and minimum-family trait have no public counterpart; Apple documents only the model framework and the convertible operation set, not the per-chip capability table or the per-operation family floor.

### Chapter 25. Compression internals {.unnumbered}

The bit-layout and address detail are M1/H13 (table-descriptor codegen version five, family two, A13) unless another version or family is named.
The sparse and int4 speedups and byte ratios, the byte-identical compiled program, and the M1 fold of the int8 and blockwise forms are M1/H13 measured; the A14 int8 stream and blockwise fold are M2/H14 (Apple M2 Pro) measured; the all-forms-stream endpoint is M5/H17s (Apple M5) measured.
The kernel-format helper tables, affine and palette dequantization codecs, dequantize-to-dense fold path, streaming and palette gates, on-chip-memory budget caps, and Winograd eligibility gate are decompile-derived, with the on-device sparse-format field read from the register map; the A15 floor at which the blockwise form first streams is predicted from the gate pattern, and the resident Winograd transform matrices are predicted, their textbook forms matching the engine's behavior but not byte-confirmable from the binary.
Apple's conversion tools document palettization, quantization, and pruning at apple.github.io/coremltools as a model-size feature; this chapter extends that account with the on-device codec arithmetic and the per-family streaming datapath.

### Chapter 26. Hidden layers and direct netplist authoring {.unnumbered}

The fused-attention, ranking, and spatial-rearrange layers were authored, compiled, and dispatched on the M5/H17s byte-exact against a host reference, and the M1 gates and the top-k forbidden band were confirmed on the M1 (measured).
The native layer-descriptor catalog, its per-layer `ZinParse<Name>Unit` parsers and `_ANECValidate<Name>Layer` checkers, and the constant-string constraint corpus are decompile-derived, joined to the netplist schema read out of the runtime framework.
On the public side, Apple documents the model converter and its intermediate-language operation set at apple.github.io/coremltools, which does not emit these native layer kinds; this chapter authors them directly, the attention, ranking, spatial-rearrange, geometry, and normalization descriptors being present in the compiler and reachable through the network description even though the converter never produces them.
The 33-knot activation-LUT format and the gated `NeuronCustom` netplist path (a parser that requires and then rejects the same field sets) are decompile-derived; the rectifier-basis reproduction of an arbitrary pointwise function is M5/H17s measured.

## Part VIII. System Internals {.unnumbered}

Part VIII rests on M1/H13 (Apple M1; M1 Max, and the T6000-generation engine where a multi-die part is needed), with an M2-class kernel cache as the cross-generation reference where cited.
Much of it is decompile-derived static analysis with no firmware executed: the unencrypted real-time-kernel preload executable is carved from the on-package firmware image and read for its strings, asserts, and disassembled handlers, and the kernel cache is read for its symbols, dispatch arrays, and call sites.

### Chapter 27. Kernel driver and IOKit ABI {.unnumbered}

The two `IOExternalMethodDispatch2022` arrays are M1/H13 measured, read byte for byte from the kernel cache's read-only data section and corroborated on an M2-class cache where all 26 size tuples are byte-identical, and the control-client open and submit struct sizes are cross-validated against captured user-space call blobs.
The selector handlers, four-layer call path, doorbell register write, entitlement-check call sites for `com.apple.ane.iokit-user-access` and `com.apple.ane.allow-dataChaining-access`, driver class hierarchy and device properties, and broker model are decompile-derived from the unstripped kernel-cache symbols, kext property lists, live device registry, and a system-wide entitlement sweep.
The driver's user-client ABI has no public counterpart and is reported as reverse-engineered; the IOKit user-client framework and the `IOExternalMethodDispatch2022` structure are public, but this driver's selector numbers, struct sizes, and handler set are not.
On the M5 the client-creation gate `ANEClientInfo::create`, its `copyClientEntitlement` stamp of `isPrivileged` and `allowDataChaining`, and six further driver-enforced `com.apple.ane` and `com.apple.private.ane` entitlements are decompile-derived from the M5 kernel driver.

### Chapter 28. Address translation and the DART {.unnumbered}

The leaf word `phys | 0x8000000000000000`, 16 KB granule, active stream set `{0, 1, 2}`, translation-table base `0x90022320`, and host-to-firmware rebase to the `0x1bc4` aperture are M1/H13 measured read-only on the live dispatch path (Apple M1 Pro, T6000-generation DART): the granule and stream set from the live device tree, the base register from function-boundary probes, and the leaf-word template, segment structure, and protection classes from probes across 26178 leaf-map events.
The leaf-word bit layout, the fault-register offset map, the panic-terminated fault path (panic confirmed M1/H13 from the disassembly of every fault-path function), the `IODARTErrorInfo` descriptor layout, the `[engine+0xe028]` status predicate, and firmware rebase arithmetic with its three aperture-config offsets are decompile-derived, the rebase arithmetic unicorn-verified.
The fault-capture register decode is predicted from the published controller field layout and not measured, because a fault panics the machine.
The address-translation unit, its leaf entry format, the rebase boundary, and the fault-capture block have no public counterpart and are reported as reverse-engineered.
The per-client isolation contexts, the eight `mapper-ane0` translation mappers in the live IORegistry and the `ANEIsoID1` through `ID7` exclave capabilities that bind them, are M5/H17s measured with System Integrity Protection enabled.

### Chapter 29. Firmware {.unnumbered}

The preload executable is the M1-generation real-time-kernel image, build identity `RTKit-3255.120.11.release`, chip tag `ASC_CHINOOK`.
The task roster, priority bands, heap and pool model, execution-loop command set, scheduler deadline, and fault post-mortem layout; the command-record classes and their `sCSneCmdProcedureCall*` invariants; the doorbell-emit sequence around bit 39 of `S3_3_C15_C8_0`, host-notify site `@0x4c890`, and engine-to-graphics-processor doorbell at `0x2_0646_8000`; the bring-up order, seven MMIO banks, RTBuddy endpoint, and three scratch handshakes; and the per-run statistics buffer with its header, per-engine descriptors, and host-side null gate are decompile-derived from the embedded strings, assertion expressions, and disassembled handlers.
The firmware has no public counterpart and is reported as reverse-engineered from the unencrypted image; Apple documents only the model framework and conversion tools above this layer, not the on-engine operating system.
The CHINOOK control-CPU register map, its eleven thread contexts, level-two cache, and pipeline error-capture and power-down-save registers, is decompile-derived from the kernel driver.

### Chapter 30. Host-to-firmware command protocol {.unnumbered}

This chapter is static analysis of the unencrypted M1 firmware image, an ARM64e real-time-kernel Mach-O, with no firmware executed.
The command vocabulary, numeric identifiers, header layout, and body bounds are read from the ordered command-name string table, struct-size asserts, and log format strings; header byte offsets are inferred from field order and alignment, while field presence, widths, and bounds are read directly from in-binary asserts.
The 94-entry `CSNE_CMD` enumeration (93 dispatched command identifiers plus the invalid sentinel), the roughly ten fast-path ids, and the seventy-six-slot dispatch vtable (arm64e auth-rebase chained pointers in `__DATA.__const`, low thirty-two bits giving the target, the procedure-call slot at `+0x200` reaching `0x7374c` and the inference slot at `+0x190` reaching `0x74510`) are decompile-derived.
The protocol, the command header, and `CSNE_CMD_*` vocabulary have no public counterpart and are reported as reverse-engineered; the public model framework describes application-level model loading, not the controller command channel. The full numeric command table and the decoded request structs are Appendix C.

### Chapter 31. Power and thermal {.unnumbered}

The clean 0 mW idle, the 176-second saturating loop holding flat power and throughput under nominal thermal pressure, the first-op power-up tax near 0.5 ms past a 100 ms idle gap, and the single held power state across 56,527 dispatches are M1/H13 measured by read-only tracing.
The power-block base `0x2_6b8f_0000`, the opaque voltage base `0x2_3b70_c008`, the five-store power-block arm, the `0x11` peak-power control word, the engine and power-manager device-tree nodes (`ane0@84000000`, `compatible "ane,t8020"`, the `0x2_8400_0000` aperture, the `0x2_8E08_0000` power-manager slice), the absence of local DVFS, and firmware seven-step credit sequence are decompile-derived from the H13 firmware Mach-O and live device-tree enumeration.
There is no public counterpart: Apple documents neither the power model, the fixed operating point, nor the thermal behavior, so this account is reported as reverse-engineered and measured.

### Chapter 32. Security and isolation {.unnumbered}

The cross-process timing side-channel (a 2.3 times latency jump under contention and a 20 to 50 bit/s occupancy channel) and the intact data isolation across 9000 concurrent results are M1/H13 measured.
The kernel-side trust boundary and its three program checks (code signature, vnode trustcache, client code-signing identity), the firmware's structural-only check, and the secure and exclave method bodies (the secure-mode transition state machine, the `SwitchExclaveMode not supported` stub, the inert `mov w0, #0; ret` exclave selectors) are decompile-derived from the loaded kernel driver and the firmware image.
There is no public counterpart: Apple documents neither the secure and exclave transition internals nor the cross-process isolation behavior, so this account is reported as reverse-engineered and measured.
On the M5 the exclave is live rather than stubbed: the secure component `com.apple.aneexclave`, its capability-scoped segment access, and the Tightbeam submit path are decompile-derived from the M5 exclave bundle and boot kernelcache, with System Integrity Protection enabled.

### Chapter 33. Telemetry and hardware counters {.unnumbered}

On the Apple M1 and M1 Max, the readable whole-engine channels (DRAM bytes, energy, clock residency), the signpost lifecycle, the all-zero per-run output buffer, and forced-mask load rejection are M1/H13 measured.
The `ANEProgramCreateArgs` layout, the `+0x6c` stats-mask offset, the twenty-four per-descriptor counter namespace, the stats-buffer ABI `0x0201`, the `initStatsBufferSection` bail branch, and the free-running engine timebase counter at MMIO `0x2_6b17_8000` (read by the firmware helper `@0x30988`) are decompile-derived from the runtime dylibs and the kernel driver.
There is no public counterpart: Apple documents none of the hardware performance-counter block, per-task-descriptor namespace, stats-mask enable, or firmware timestamp, so the block geometry, master enable, readable-versus-walled split, and kernel gate are reported as reverse-engineered and measured.
On the M5 the gate actor (the `aned` daemon forcing `statsMask=0` for a `ThirdPartyAppUsingANE` client), the per-channel `DCS BW / ANE L0` and `L1` State-residency bandwidth histograms readable on the unentitled path, and the fuller `ANE_THROTTLE_*` and `VDD_DRAM_VOLTAGE_CHANGE` trigger family are M5/H17s measured with System Integrity Protection enabled.

## Part IX. Cross-Silicon Reference {.unnumbered}

Part IX was measured on M1/H13 (Apple M1; M1 Max) with M5/H17s (Apple M5) as the cross-chip reference; the work is decompilation and static analysis of the one engine compiler binary across its full target set, with boundaries reproduced on silicon where a part was in hand.

### Chapter 34. Cross-silicon targets {.unnumbered}

The 28-target set is measured, extracted by invoking each per-architecture builder on the M1 (host chip irrelevant) and resolving the M5 to `H17s` and the M1 Max to `H13G` by fixed-build-directory compile.
The target names, the suffix-to-core decode at HAL offset `0x238`, the interchange-format tables at HAL offset `0x658`, and four-byte format decode are decompile-derived; the A14, A15, A16, and A18 targets and their M-series counterparts are decompile-derived from the per-target tables and not individually measured, and the $M(n) \rightarrow H(n+12)$ mapping is confirmed only at the M1 and M5 ends with the middle generations predicted.
On the public side, Apple's product specifications publish a marketing core count per chip, for example a 16-core engine, a different quantity from the decoded `num_nes`, which counts per-die compute sets: four on the base M1 against the published sixteen.
The 28-target compiler set, the H-architecture naming, the suffix-to-core decode, and interchange tables are not publicly documented and are reported as reverse-engineered.

### Chapter 35. Per-family code generation {.unnumbered}

The slice-saturation threshold, the top-k, sort, and dynamic-slice code-generation rejections, and the operation decompositions are M1/H13 measured; the clean slice route and the native crop-resize, resample, and trig operations are M5/H17s measured.
The family enum, `MinimumFamily<N>` trait and its four operation tiers, per-chip hardware-abstraction offsets, task-descriptor patch-width path, and `ConvertSlice<Family>` lowering are decompile-derived from the engine compiler framework (the 9.509 build, 87,874 functions); the A14 and A15 unlock points, M-series families above the M1, and dedicated A15 code-generation branch with its cost table, 45-operation floor, full H15 targets, and YUV420 input are decompile-derived and not measured on A15 silicon.
The family enum, minimum-family trait, per-chip parameters, and per-family route selection have no public counterpart; the public conversion tools document the frontend operation set, optimization passes, and compute-unit selector, not the backend per-family lowering reported here.

### Chapter 36. Predicted upper tier {.unnumbered}

The fp8 format converters, E4M3 overflow enumeration, format-register encoders, double-multiply gate, collective dialect operation set, device-mesh and sharding lowering, reduction-to-atomic map, and collective direct-memory-access emitter are decompile-derived, with the family gates read from the 28-target capability bytes; the 64-core ceiling, fp8 `e4m3` and `e5m2` datapath, and Ultra device-mesh collective are decompile-derived and not measured on the upper-tier parts.
The E4M3 native multiply, accumulation, and saturation (inferred from the encoders and the H18-only capability byte at offset `0x52d`), the fp16 accumulation of an fp8 multiply (from the absence of any fp8 accumulator field), and the running collective (the enable byte at offset `0x48b` zero on all 28 targets and the register encoding stubbed on every family) are predicted, with no current family materializing the collective.
That the load-balancer supports up to four engine dies while the M1 and M1 Max each register a single engine, so cross-die steering engages only on a multi-die part such as the Ultra, is M1 Max measured by the device registry.
The fp8 datapath and the multi-die collective layer have no public counterpart in Apple's documentation and are reported as reverse-engineered and explicitly unmeasured.

## Back matter {.unnumbered}

The back-matter chapters rest on M1/H13 as the primary host and M5/H17s for cross-generation scaling.

### Methodology {.unnumbered}

Apple documents the engine only as a compute-unit selector at developer.apple.com/documentation/coreml/mlcomputeunits, with no direct device API; this chapter extends that account by reaching the engine directly below the selector and characterizing it by static analysis and live instrumentation.
The direct dispatch route, the four static-analysis artifacts, and the program-binary capture are decompile-derived and confirmed by the kernel trace; the roofline figures and the compile-service rate condition are M1/H13 measured, and the cross-generation scaling and bounded numeric drift are M5/H17s measured.

### Open questions {.unnumbered}

The decoded baseline and the boundary limits are M1/H13, decompile-derived from the compiler, hardware-abstraction tables, kernel driver, and firmware and joined to live instrumentation; M5/H17s confirmed the cross-family predictions.
The M3/H15 and upper-tier runtime behavior is predicted, decompile-derived from the gates and not confirmed on silicon.

## Appendices {.unnumbered}

The reference tables are decompile-derived from static, read-only analysis of the M1/H13 binaries: the ANE compiler (`ANECompiler 9.509`), its per-family operation-floor tables, its per-layer validators and parsers, the host runtime, and the unencrypted firmware image (`h13_ane_fw_styx_j5x.im4p`), with no firmware executed and no engine jobs run.
Where a value or status is measured rather than decompile-derived, it was confirmed on physical silicon, primarily M5/H17s and M1/H13, by compiling and dispatching against a host reference.

### Appendix A. The operation-by-device matrix {.unnumbered}

The M1, M2, and M5 columns are measured by operation-conformance runs, each native operation compiled and run and each no-path one rejected on device; the M3 column and the M4 part of the merged M4-and-M5 column are decompile-derived predictions from the per-chip tables.
The per-family unlock points for the texture-engine operations, sin and cos, rank and sort bridge, argument reductions, and weight-stream gates are decompile-derived from the operation floors, validators, and symbol-resolution map and confirmed on the M1 only at its boundary; the family that first runs each is predicted from the floor table.
Apple documents the convertible operation set at apple.github.io/coremltools and developer.apple.com; this table extends and partly corrects that account, reporting what compiles and runs on the direct engine path, since some accepted operations, such as three-dimensional convolution, do not lower to the engine.

### Appendix B. The hidden-layer catalog {.unnumbered}

Each layer's `Type` tag, descriptor symbol, and `Params` key set are decompile-derived from the compiler export table, parser disassembly, constant-string key atlas, per-layer `ZinParse<Name>Unit` parsers and `_ANECValidate<Name>Layer` checkers, and descriptor-initializer routines `_ANEC<Name>LayerDescInitialize`, joined to the netplist schema read out of the runtime framework.
The fused-attention operand contract, the spatial-rearrange channel ordering, the float16-bit-pattern convention for `Alpha`, `Epsilon`, and `Scale`, and point-cloud output contracts are M5/H17s measured by authoring and dispatching the layers, and the M1 arch gates, the `Sort` and `DynamicSlice` rejections, and the top-k $\{3, 4\}$ forbidden band are M1/H13 measured.
Apple documents the model converter and its intermediate-language operation set at apple.github.io/coremltools, which does not emit these native layer kinds; this catalog authors the native descriptors directly through the network description.

### Appendix C. Decoded reference tables {.unnumbered}

Every value is read out of an M1/H13 binary by static analysis, with no firmware executed: the attribute and opcode integers and IOKit struct layouts from the compiler decompile (`ANECompiler 9.509`) and the host runtime; the error constants, command table, and tunable table from the unencrypted firmware image (`h13_ane_fw_styx_j5x.im4p`) and the standard IOKit return macros; the register map from the compiler's task-descriptor setters and getters; and the `.e5` schema from the runtime serializer symbols, validated byte-for-byte against a captured sample.
The runtime, firmware, compiler, and ABI surface consolidated here is private and undocumented and is reported as reverse-engineered; the public model framework, conversion tools, and intermediate-language reference describe none of these numeric tables.
