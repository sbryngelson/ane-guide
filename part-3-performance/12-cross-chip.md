# 12. Across the chip family

> An M(n) chip has the H(n+12) ANE architecture, so M1 is H13 and M5 is H17.
> A network that compiles and runs on one generation compiles and runs on the others, because one compiler binary builds every target and only a per-target data table changes.
> The single property that varies from chip to chip is fp16 numerics, and it varies only at a unit in the last place.
> Target the generation a network needs by its operation set, then verify per chip only the cancellation-sensitive reductions and the width-axis slices.

## Naming rule

The M-series engine and the contemporaneous A-series engine are the same architecture under two product names, offset by a fixed amount.
An M(n) chip has the H(n+12) ANE architecture, the compact relation $M(n) \rightarrow H(n+12)$.
M1 is H13, M2 is H14, M3 is H15, M4 is H16, and M5 is H17.
The A-series anchor is one generation over: the A13 and the M1 share H13, and the A17 and the M5 share H17.

[Table](#tbl:c12-family) is the family map of the core count and clock that scale across the generations, with the un-measured upper generations decompile-derived from the device tables.

| Chip | A-series anchor | ANE architecture | NE cores | Clock |
| --- | --- | --- | --- | --- |
| M1 | A13 | H13 | 4 (base), 8 (Pro and Max) | ~1.14 GHz |
| M2 | A14 | H14g | 4 (base), 8 (Pro and Max), 32 (Max-class) | measured |
| M3 | A15 | H15 | 4 (base), predicted | predicted |
| M4 | A16 | H16 | 4 (base), predicted | predicted |
| M5 | A17 | H17s | 16 | ~1.89 GHz |

Table: The M-series chips with their A-series anchor, engine architecture string, core count, and clock. {#tbl:c12-family}

Three rows are measured on physical silicon: M1 (H13), M2 (a Pro reporting H14g), and M5 (H17s); the M3 and M4 rows are decompile-derived from the per-family device tables and not individually measured, with the A15/M3 generation the one rail that remains unmeasured.
The M2 measurement closes the middle of the sequence: a seeded classifier trains to the M1 number to the digit, the four fp16 axes match the M1 bit for bit, and a watt-complete device map reproduces the M1 and M5 shape.
The A14 is thus the A13's numerical twin on every axis measured.
The fuller silicon-to-target table, the board-type sequence, and the full set of 28 compiler targets are the subject of a later chapter [AppleANE].

The rule reads directly off the device tables and is confirmed on the measured parts.
The live M1 reports the architecture string h13g, the Pro and Max variant of H13, and the M5 compiles to the H17 target and resolves the H17s variant on disk.
The system frameworks corroborate the sequence: the on-device video upscaler includes exactly the five targets H13 through H17, which are the five Mac generations M1 through M5.
The target a network needs can thus be named by generation, and the mapping holds without probing the silicon.

## What holds across the family

A network that compiles and runs on one generation compiles and runs on the others.
The compiler is a single binary that constructs any target on demand, so the program format, operation legality, and datapath are shared, and only a per-target data table changes underneath them.

The operation limits are properties of the family, not of one chip.
The operations with no hardware path on any current part, such as the product reduction, scatter family, and recurrent cells, are absent on every generation from the M1 through the M5.
Gated operations arrive at a known generation and stay: the texture-engine sampler operations turn on at the A14, and native sin and cos turn on at the A15.
A network that avoids them runs everywhere, and a network that uses them runs from its unlock generation forward.
No operation the M1 can compute is missing on the M5, because each newer engine is an operation superset of the one before it.

Each generation adds one capability over the one before it, then stops, and [Table](#tbl:c12-caps) names the single capability each engine generation adds, read off the per-target capability bytes.

| Generation | Capability added |
| --- | --- |
| A13 (M1) | three-dimensional convolution, the sixteen-deep kernel, native softmax, layer norm, all reductions, and fused attention |
| A14 (M2) | the texture-engine samplers (resize, crop-resize, resample, affine, hardware gather) and cross-die addressing |
| A15 (M3) | native sin and cos, the dropout and random path, and global argument-min and argument-max |
| A16 (M4) | the tensor dimension limit rises from 16384 to 65536, and the fp16 kernel-width ceiling rises from 13 to 15 |
| A17 (M5) | no operation over A16; the NE-core count scales |

Table: The single capability each engine generation adds, read off the per-target capability bytes. {#tbl:c12-caps}

The A17 and A18 add no operation over the A16: identical dimension limits, the same texture engine, the same legal operation set, differing only in NE-core count, which scales throughput rather than legality.
The dimension limit is not a single number per chip: on the M2 the spatial and contraction extents cap at 16384 while the channel axis caps at 65536, exactly four times the spatial cap.
The limit thus belongs to the axis an operation uses rather than to the tensor.

The newer parts scale the core count and the clock; they do not change the programming model.
The core count runs 4 on the M1, 8 on its Pro and Max variant, and 16 on the M5, and the operating clock rises from roughly 1.14 GHz to roughly 1.89 GHz across that span.
The fp16 datapath, the wide accumulator, and the form of the roofline hold across the family unchanged.
The measured M5 confirms the scaling: about 19.6 fp16 TFLOP/s on the matmul slope and about 14.3 fp16 TFLOP/s on the convolution peak, both from the fused-chain probe on the same network with no source change, against the roofline saturation peak of 18.8 TFLOP/s in Chapter 9.
A single large matmul above the dispatch floor runs at about 9.5 fp16 TFLOP/s, the M5 analogue of the M1's 4.8, and the engine streams weights at about 145 GB/s over two DRAM read channels, near three times the M1's 51 GB/s.
Those peaks are set by the larger core count and the higher clock, not by any change to the datapath, which is the metric on which the generations compare.
The working-set threshold moves with the silicon, from near 2 MB on the M1 to a measured 4.72 MB on the M5, scaling with the larger 16-core on-chip memory.

## One thing that varies

The single property that differs from chip to chip is fp16 numerics.
The accumulator width is uniform across every engine and the compiler text is identical, so a cross-chip value difference can only come from a data-selected codegen route that changes the order in which fp16 operations combine.
That surface is limited: most route changes are a numerical no-op, because the wide accumulator absorbs the reordering, and the rest are at a unit in the last place, set by tiling-boundary alignment.

The cross-generation measurement fixes the scale.
The same seeded convolutional classifier trained on the M1, M2, and M5 reaches 0.9080, 0.9080, and 0.9070 test accuracy, each deterministic across repeated runs, a difference of one test sample in a thousand between the ends.
The M2 is exactly on the M1 number, which puts the entire fp16 training drift at the A16 generation rather than spread across the family: the M1 and the M2 are numerical twins, and the gap opens only at the M5.
That gap is the drift of sub-unit-in-the-last-place fp16 differences compounding over a few hundred training steps, real and negligible.
The cross-silicon predictions extracted from the device tables were confirmed on the M5: all ten, covering throughput, the working-set threshold, operation limits, texture engine, and fp16 slice behavior, held on the real part.
The one finite-to-infinity axis, a slice saturation that occurs on the M1, takes the non-saturating route on the M5, as predicted.

## fp16 divergence axes

That data-selected codegen surface reduces to four axes.
Three of them are at most a unit in the last place, and one is a finite-to-infinity saturation that occurs on the older parts.
[Table](#tbl:c12-axes) names the four axes, the codegen route each selects, and the bounded magnitude of each.

| Axis | Mechanism | Effect | Magnitude |
| --- | --- | --- | --- |
| Slice saturation | A width-axis slice with a nonzero offset routes through a fixed-point crop that multiplies by sixteen | A source value above 4094 saturates to plus or minus infinity, since 4094 times sixteen is the 65504 fp16 ceiling | finite to infinity |
| Reduction then square fusion | A reduction immediately followed by a square or multiply can fuse, removing one intermediate rounding step | Drops one fp16 rounding step | at most one unit in the last place |
| Reduction route | A reduction selects a transpose route or a reshape route by an extent threshold, 192 on the older parts and 384 from the M3 | Reorders partial sums | numerical no-op, the wide accumulator absorbs it |
| Tiling granularity | The partial-sum tile alignment is set by the patch and core-count fields, granularity near 128 | A sum off a tile boundary loses one rounding increment | at most one unit in the last place |

Table: The four fp16 cross-chip divergence axes, the codegen route each selects, and the bounded magnitude of each. {#tbl:c12-axes}

The saturation axis is the only one that changes a finite value into an infinity, and it is magnitude-gated: it triggers only when a width-offset slice holds a value above 4094.
Measured on the M1 the threshold is exact: a width-offset slice is finite at 4094 and goes to plus or minus infinity at 4100, while the zero-offset control stays finite even at 60000.
The M2 saturates bit-identically to the M1, which corrects the earlier reading that the non-saturating route arrives at the A14: the saturation persists through the A14 and the M5 is the part that takes the non-saturating route.
The reduction-then-square fusion never manifests on silicon: the M1, M2, and M5 all measure the unfused result, so that axis is uniform across the family.
A divergence is thus predictable from a small set of fields without running every chip: a reduction or normalization denominator can differ by at most one unit in the last place where the tiling or route fields differ.
A width-offset slice has saturation risk only when its values can exceed 4094.

## Developer policy

Choose the generation a network requires from its operation set, then let the program run across every part at and above that generation.
Verify per chip only the numerics that can move: the cancellation-sensitive reductions, the variance and normalization denominators, and any width-axis slice whose values can exceed the saturation bound.
Everything else is portable by construction: the same source, the same operation legality, faster on the newer silicon by the core and clock scaling.

## Compiling for a target generation

The naming rule lets a network name its target by generation rather than by probing the silicon.
The compiler constructs any target from its per-chip table, so a developer compiles a program for the oldest generation it must support, then runs it unchanged on every part at and above that generation.
A static estimate against a target reads back that target's core-and-clock scaling without the part in hand.

The procedure compiles for the floor generation a network requires, then estimates against the newer targets to read the core-and-clock speedup.

```c
/* The target generation is held in the compiler options as a TargetArchitecture string, */
/* so the same source compiles for whatever floor the network must support.            */
e5rt_e5_compiler_options_create(&options);
e5rt_e5_compiler_options_set_custom_ane_compiler_options(options, "TargetArchitecture=h13");
e5rt_e5_compiler_compile(compiler, model_path, options, &library);  /* M1 floor, runs M1..M5 */

/* Then the same drive: retain the function, build the op, and dispatch on a stream. */
e5rt_program_library_retain_program_function(library, fn_name, &function);
e5rt_precompiled_compute_op_create_options_create_with_program_function(function, &op_opts);
e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options(op_opts, &op);
e5rt_execution_stream_encode_operation(stream, op);
e5rt_execution_stream_execute_sync(stream);
```

A network compiled to `h13` runs on every generation from the M1 through the M5; estimating the same graph against `h17s` reads back the M5 scaling without the M5 in hand.

## Reference: per-family scaling constants

[Table](#tbl:c12-constants) collects the per-family scaling constants and the figures that fix cross-generation behavior, with the silicon each was measured on.

| Quantity | Value | Silicon |
| --- | --- | --- |
| Naming rule | $M(n) \rightarrow H(n+12)$ | family-wide |
| NE cores | 4 (base), 8 (Pro and Max) | M1/H13 |
| NE cores | 16 | M5/H17s |
| Operating clock | ~1.14 GHz | M1/H13 |
| Operating clock | ~1.89 GHz | M5/H17s |
| Matmul-slope peak | about 19.6 fp16 TFLOP/s | M5/H17s |
| Single-program matmul peak | about 9.5 fp16 TFLOP/s | M5/H17s |
| Convolution peak | about 14.3 fp16 TFLOP/s | M5/H17s |
| M5 convolution peak versus M1 projected fp16 peak | near 5x | M5/H17s |
| Working-set threshold | near 2 MB | M1/H13 |
| Working-set threshold | 4.72 MB | M5/H17s |
| DRAM weight-stream bandwidth | about 145 GB/s, two read channels | M5/H17s |
| Texture-engine sampler unlock | A14 generation | family-wide |
| Native sin and cos unlock | A15 generation | family-wide |
| Tensor dimension limit raised, 16384 to 65536 | A16 generation | family-wide |
| Kernel-width ceiling raised, 13 to 15 | A16 generation | family-wide |
| M2 spatial and contraction extent cap | 16384 | M2/H14 |
| M2 channel-axis extent cap | 65536 | M2/H14 |
| Cross-generation training parity | 0.9080, 0.9080, 0.9070 | M1/H13, M2/H14, M5/H17s |
| fp16 cross-chip divergence bound | one unit in the last place | family-wide |
| fp16 slice-saturation threshold | source above 4094 to infinity | M1/H13, M2/H14 |
| Number of fp16 divergence axes | four | family-wide |
| Cross-silicon prediction pass | ten of ten | M5/H17s |

Table: The per-family scaling constants and the figures that fix cross-generation behavior, with the silicon each was measured on. {#tbl:c12-constants}

Part IV turns from the engine in isolation to the workloads that run on it.
