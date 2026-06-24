# 35. Per-family code generation

> One compiler binary lowers a network for every generation, and the per-chip difference is data rather than code.
> Native operation support is declared by a `MinimumFamily<N>` trait in four floors, F0 through F4, and nothing floors above F4.
> The only operations the M1 decomposes that the M5 runs natively are crop-resize, resample, `sin`, `cos`, and the global arg-reductions.

The compiler that targets every Neural Engine generation is one binary.
A single build lowers a network for the M1 and for the M5, and the divergence is driven by the family enum and the per-chip hardware-abstraction parameter blob from chapter 24.

## One binary, eight families

The compiler enumerates the eight generations as the `mlir::anec::Family` values that [listing](#lst:c35-family) gives.

```cpp
mlir::anec::Family = {
  A11Legacy = 0, A12 = 1, A13 = 2, A14 = 3,
  A15 = 4, A16 = 5, A17 = 6, A18 = 7
}
```

Listing: The eight-value family enumeration the compiler keys per-generation operation legality from. {#lst:c35-family}

The M1 compiles as `A13` (index 2) and the M5 compiles as `A17` (index 6), and `A17` is a strict superset of `A13` in operation support.
Two string parsers build the per-target objects from the target name: `CreateTargetFromString` constructs the `ZinIrTarget` instance, and `CreateHalFromString` constructs the matching `ZinIrHalParameters` blob.
Both branch on string length and the first characters compared as little-endian shorts, `0x3168` for "h1", `0x3074` for "t0", `0x396d` for "m9", and `0x316d` for "m1".

The numeric hardware version maps to the family enum through a fixed table, the `unordered_map<int, Family>` named `ANEFamilyToMLIR`, whose data table [table](#tbl:c35-fammap) decodes key to value along with the targets that resolve to each family.

| Hardware version | Family index | Targets |
| --- | --- | --- |
| 1 | A11Legacy (0) | `h11` |
| 3 | A12 (1) | `h12` |
| 4 | A13 (2) | `h13`, `h13g`, `t1` |
| 5 | A14 (3) | `h14`, `h14g`, `h14c` |
| 6 | A15 (4) | `h15`, `h15g`, `h15c` |
| 8 | A15 (4) | `h16`, `h16g`, `h16c`, `h16s` |
| 7 | A16 (5) | `h17`, `h17a`, `h17g`, `h17c`, `h17d` |
| 9 | A17 (6) | `h17s` |
| 10 | A17 (6) | `h18` |

Table: The hardware-version-to-family map read from the `ANEFamilyToMLIR` data table, with the targets that resolve to each family. {#tbl:c35-fammap}

This table holds two collisions that the compiler must guard.
Hardware versions 6 and 8 both resolve to family A15, so `h16` and the A16-tier parts compile with A15-family operation legality.
Hardware versions 9 and 10 both resolve to family A17, so `h18` uses A17-family operation legality.
The family enum drives the `MinimumFamily<N>` operation legality only.
The per-target hardware-abstraction blob drives code generation, the numeric limits, and the cost model.
The suffixed targets such as `h17s` thus select a distinct `ZinIrTarget` object with its own descriptor generation, core count, and cost curves, even when several family-keyed paths collapse onto the same enum value.
The hardware-version-to-family table is read directly from binary data; the per-target hardware-version value has some inference, since the per-target hardware-version getter is virtual and not present in the readable decompilation.
The two anchors `h13` to A13 and `h17s` to A17 are confirmed by the live measurement campaign.

The lowering code is shared, so the per-family split is in two locations.
An operation trait declares a minimum family for native support.
The per-chip hardware-abstraction blob is a fixed-size per-target structure whose field values, not whose code, select the route a shared lowering pattern takes.

## Operation floors: the minimum-family trait

[Table](#tbl:c35-floors) gives the four minimum-family floors, the families each is native on, and representative operations at each floor.

| Floor | Native on | Representative operations |
| --- | --- | --- |
| F0 | all 8 families | convolution, deconvolution, matmul, pooling, all elementwise, reshape, transpose, concat, sigmoid, tanh, relu, gelu, swish, quantize, dequantize |
| F2 (A13+) | A13 through A18 | softmax, layer-norm, instance-norm, batch-norm, all reductions, resize, scaled dot-product attention, `erf`, `exp2`, `rsqrt`, `sqrt`, tile, space-to-channel |
| F3 (A14+) | A14 through A18 | crop-resize, resample |
| F4 (A15+) | A15 through A18 | `sin`, `cos`, global arg-min and arg-max |

Table: The four minimum-family floors, the families each is native on, and representative operations at each floor. {#tbl:c35-floors}

Native operation support is declared by the `MinimumFamily<N>` trait held on each backend operation, the operation-side member of the two capability gates chapter 24 defines.
An operation is natively legal only inside a region whose family index is at least `N`; below that floor the compiler decomposes it into a sequence of supported operations.
The trait is the mechanism, not the dynamic-legality registration: `addDynamicallyLegalOp` wraps only the eight region operations and four texture-gated operations, while every backend operation has the `MinimumFamily<N>` impl.
88 compute operations have the trait, 52 at F0, 31 at F2, two at F3, and three at F4.
These operation floors are measured on physical silicon for the M1 and M5; the M3, M4, and upper-tier figures are decompile-derived, predicted from the per-chip tables rather than measured.

No operation in the decoded compute-op floor table floors above F4, so almost nothing is exclusive to `A16`, `A17`, or `A18`: the upper-tier generations add Neural Engine cores rather than a usable operation set.
The one exception is a small set of `MinimumFamily<N>` template instantiations at N=5, N=6, and N=7, including the fp8-bearing operations, which stay inert below H18 (chapter 36).
Softmax, layer-norm, all reductions, resize, and scaled dot-product attention floor at `A13`, so they run natively on the M1.

The two oldest families behave as a different, more constrained instruction set.
On `A11Legacy` and `A12` there is no native reshape, broadcast, or divide.
Reshape, squeeze, expand-dims, and broadcast all share `ConvertToReshape<Op, Family>` and `ConvertBroadcast<Family>`.
On A13 and above they emit a plain native `anec::Reshape` or `anec::Broadcast`, while on the two legacy families reshape lowers to `anec::Flatten` and the compile aborts through `verifyCompatibilityWithFlatten` when the layout does not collapse to a pure flatten.
On the two legacy families squeeze or expand-dims emit the string "cannot be lowered as Flatten on ANE".
Broadcast on the legacy families asserts fp16 and switches engine by axis: a channel-group axis broadcasts as a matrix multiply by ones, and other axes broadcast as an elementwise add by zero.
Divide on the legacy families matches only a constant fp16 divisor and lowers as a reciprocal multiply with the reciprocal pre-rounded to fp16, with a non-constant or non-fp16 divisor producing a match failure.
The floor-divide form is $\lfloor a \cdot \mathrm{f16}(1/b) \rfloor$.
The double rounding can give the wrong integer at a boundary, for example $\lfloor 6 \cdot \mathrm{f16}(1/3) \rfloor = \lfloor 1.999 \rfloor = 1$.
On A13 and above divide is a native `anec::ElementwiseDiv` over arbitrary dynamic divisors.

## Same operation, a different kernel

[Table](#tbl:c35-kernels) gives the five categories where the shared lowering emits different machine code or different numbers per chip, with the per-chip parameter that drives each.

| Category | Divergence | Driven by |
| --- | --- | --- |
| width slice | M1 saturates at the Q.4 crop route, M5 stays in fp16 | patch-width clamps, width granule |
| resize | three strategies round differently by chip | texture-engine-present flag |
| matmul | lowers to a resident convolution, splits along height | per-family copy-cast and max-tensor-size thresholds |
| transpose | native within a height-width bound, else multi-pass | per-chip maximum height and width |
| strided convolution | factorizes stride into a sub-convolution sequence | per-chip allowed-factor list |

Table: The five categories where the shared lowering emits different machine code or different numbers per chip, with the per-chip parameter that drives each. {#tbl:c35-kernels}

The lowering pattern is byte-identical across families in each case, and the divergence is in the hardware-abstraction parameters or the task-descriptor emitter.
Take first the slice that saturates on the M1.
The trigger is `ZinSliceLayer::SliceNeedsCropMode`, which is family-independent.
A slice with a nonzero offset on the height axis (axis 3) or the width axis (axis 4) routes through crop mode to `ZinTECropModeLayer`, the transpose-engine crop direct-memory-access path, while a begin of zero and the non-height-width axes skip it.
The height axis decompiles to the same path but was measured unaffected on silicon, and a channel or batch offset stays unaffected as well: only the width offset triggers the saturation.
The conversion rewrites `ConvertSlice<Family>` and `ConvertStridedSlice<Family>` are byte-identical across all eight families, and the crop scale is still a float at the layer-configuration stage where `Create` builds `1.0/(extent - 1)`.
The multiply by sixteen appears only when that geometry is written into the task descriptor.
The patch width is a four-bit field that `SetPatchWidth` writes with the instruction `bfxil w9, w8, #0, #4`, and `IsFormatDMAConvertibleToFP16` and `L2Allocate::ConvertFmt` make the fixed-point-versus-fp16 choice, keyed to the format rather than to the chip.
On the M1 the compiler sends that copy through a fixed-point storage format with four fractional bits, an implied multiply by sixteen.
The stored value is the source times sixteen and the storage clamps at the fp16 maximum, so any source magnitude above the threshold overflows to $\pm\infty$.
The threshold is exact:

$$V_{\max} = \frac{65504}{16} = 4094$$

`ZinMirL2Config::CalculateDMASrcBufferSizeAndStrides` reads the width granule and clamps, taking a `ZinIrHalParameters const&` and reading `HAL+0x1c0` as both a divisor that recovers an element count and a multiplier on the final byte stride.
It also reads the patch-width clamps at `HAL+0x3f8`, `HAL+0x400`, and `HAL+0x410` through `ComputeMaxPatchWidth`, which returns a `CeilLog2(width)` bounded by those clamps.
A sixteen-wide granule gives a log-base-two of four, the shift-left-by-four that is the multiply by sixteen.
On the M5 the route avoids the fixed-point format and the slice stays in plain fp16, so the same slice that saturates on the M1 is unaffected on the M5.
The lowering pattern is the same template on both chips; the route and format selection at the descriptor and hardware-abstraction layer differs, driven by the patch-width clamps and the width granule in the per-chip blob.
The saturating path is present on every patch-capable descriptor generation including the A17 generation.
The M1-versus-M5 difference is the route and format selection upstream, not a per-family rewrite.
Any height-width-axis slice with a nonzero offset whose fp16-convertible source can exceed 4094 is thus hazardous on the whole patch-capable generation set, unless the route is known to avoid crop mode.

The remaining four cases are decompositions gated by the per-chip blob rather than the family enum.
Resize is family-uniform at the conversion level and diverges in `ZinResizeLayerUtils`: `DecomposeResize` branches on the texture-engine-present flag at `HAL[0x81d]`, and if it is absent and the no-texture decomposition fails the compiler asserts "failed to map resize layer on this arch".
The no-texture path picks a deconvolution upsample against a transpose-plus-convolution path through `QualifiesforUpsampleDecomposition`, gated on the one-by-one fast-path flag at `HAL[0x812]`, and `GetMaxSmallKernelWidth`; the three strategies round differently, so resize results differ by chip.
Matmul lowers to a resident convolution through `LowerNEMatMulToNEConv` when the right-hand weight bytes meet the per-family copy-cast threshold at `HAL[0x1b8]`, and `SliceMatMulAlongHeight` splits the operation by the per-family maximum tensor size.
Transpose runs natively only when `IsValidNETransposeConfiguration` passes the maximum height and width at `HAL+0x468` and `HAL+0x470` plus the divisibility checks, and otherwise decomposes into multiple passes on the more constrained families.
A convolution with a large stride factorizes the stride through `DecomposeConvWithLargeStride` against the per-family allowed-factor list at `HAL[0x730]` through `HAL[0x750]`, at most two factors per axis, so one convolution becomes a different sub-convolution sequence per chip.

## What the M1 blocks and which chip enables it

[Table](#tbl:c35-blocks) gives the operations and behaviors the M1 blocks, the first Apple family that enables each, and the cause.

| Operation or behavior | M1 / H13 | Enabling family | Cause |
| --- | --- | --- | --- |
| `sin`, `cos` native | decomposed | A15 (M3+) | F4 trig floor; M1 uses a polynomial decomposition |
| crop-resize, resample, affine, height-width gather | decomposed | A14 (M2+) | texture engine absent (HAL flag `0x81d`) |
| top-k, sort, dynamic-slice | rejected at code generation | A14 (M2+) | rank and sort bridge; passes the validator, fails lowering |
| dropout, random | unavailable | A15 (M3+) | random-number block off (HAL `0x4a9`) |
| global arg-min, arg-max | decomposed | A15 (M3+) | HAL `0x4f2` |
| reduce-then-square fusion | unfused form emitted | A14 (M2+) | a fusion, not a capability; result within one unit in the last place |
| width slice with offset over 4094 | saturates to `±inf` | A15 (M3+) | M1 routes through the Q.4 crop direct-memory-access; A14 saturates too, the clean route arrives on A15+ |
| int8-affine weight streaming | folds to fp16 | A14 (M2+) | streaming gate off on `A13`; the M1 streams int4-LUT and sparse only, the M2 adds int8 |
| blockwise weight streaming | folds to fp16 | A15 (M3+) | folds on `A13` and `A14`; broad streaming arrives on A15 |

Table: The operations and behaviors the M1 blocks, first Apple family that enables each, and cause; the mapping of family to silicon is the one given in chapter 34. {#tbl:c35-blocks}

Some operations have no native form on any family, including `tan`, `asin`, `acos`, `atan2`, `sinh`, `cosh`, `atanh`, `asinh`, `acosh`, the logical and-or-xor, the recurrent cells, scatter, one-hot, non-zero, `mod`, band-part, and reverse-sequence.
These are decomposed on the host or in the graph on every chip and are not a per-family difference.
The slice saturation is the only axis that turns a finite value into an infinity across chips; the wide accumulator and the reduction route are uniform across families and are not a block.

## Kernel-data path: per-core replication against shared kernel memory

The weight layout splits into two code-generation functions that the kernel-memory register at `HAL[0x48f]` selects.
`ZinMirBuildNEKernelData` is the per-core replicated path: on the M1 the kernel coefficients are built per Neural-Engine core, walked through the per-core byte stride that `CalculateNEOffsetJump` computes, and must fit the 64 KB kernel-memory cap.
`ZinMirBuildNEKernelDataSharedKmem` is the A14-and-above shared path, looped over core groups, selected by the kernel-memory register-select byte and the `ExceedKmemSizeLimit` logic that [listing](#lst:c35-kmem) gives.

```c
off = (useShared && HAL[0x48f]) ? 0x210 : 0x200
```

Listing: The kernel-memory offset selection between the shared 16 MB path and the 64 KB per-core path. {#lst:c35-kmem}

On the M1 the register-select byte at `HAL[0x48f]` is set, but the extended-kernel-memory-mode scalar at `0x288` reads zero on every target in this compiler.
The shared 16 MB path at offset `0x210` is thus taken only when an operation explicitly requests shared and the chip enables it.
Otherwise the M1 falls to the 64 KB cap at offset `0x200` and tiles.
The A14 and later chips route the large kernels through the shared builder instead of replicating per core, which is the concrete kernel-memory architecture difference between the M1 and the later families.

## Task-descriptor layer: per-generation field offsets

Below the family enum is the descriptor emitter `ZinAneTd<Nu>`, instantiated for the generations $\{1, 4, 5, 6, 7, 8, 10, 11, 17, 19, 20\}$.
Only the generations $\{7, 8, 10, 11, 17, 19, 20\}$ have the patch-width path; the legacy generations $\{1, 4, 5, 6\}$ use `HandleCcdmaLayer` with no patch settings.
The same logical field is at a different byte offset per generation, so a field written at the wrong offset for the target corrupts an unrelated descriptor word.
[Table](#tbl:c35-descoffset) gives the patch-width descriptor offset per task-descriptor generation.

| Descriptor generation | Family | Patch-width offset |
| --- | --- | --- |
| 7 | A13, M1 | `desc+0x200` |
| 8 | A13, A14 | `desc+0x224` |
| 10 | A14 | `desc+0x12c` |
| 11 | A14, A15 | `desc+0x21c` |
| 17 | A15, A16 | `desc+0x23c` |
| 19 | A16, A17 | `desc+0x248` |
| 20 | A17, M5 | `desc+0x26c` |

Table: The patch-width descriptor offset per task-descriptor generation. {#tbl:c35-descoffset}

The primary and secondary source direct-memory-access fields and the result direct-memory-access field follow the same per-generation-offset pattern.
A single compiler defect is in this layer.
`SetPatchSettings<7u>` targets `ZinAneTd<7u>` on every setter except the last, which calls `ZinAneTd<8u>::SetTileOverlapPadReflect` on a generation-7 pointer, writing the reflect flag at the generation-8 offset; a shipping target that uses generation 7 locates its tile-overlap-pad-reflect bit in the wrong word.

## Hardware-abstraction parameter field map

`ZinIrHalParameters` is a 0x348-byte per-chip blob, copied by a 0x348-byte copy constructor.
[Table](#tbl:c35-halmap) gives the fields the shared lowering reads to drive the per-chip route, format, and limit selection.

| Offset | Meaning |
| --- | --- |
| `0x1b8` | matmul right-hand copy-cast byte threshold |
| `0x1c0` | large-stride-conv enable and direct-memory-access width granule |
| `0x328` | conv stride and unicast input-channel limit |
| `0x30`, `0x38`, `0x48`, `0x50` | maximum small and large kernel width by tensor format |
| `0x88`, `0x90`, `0x98`, `0x138` through `0x1a8` | maximum spatial and per-dimension tensor sizes |
| `0x468`, `0x470` | maximum transpose height and width |
| `0x730` through `0x750` | allowed kernel-width and stride factor list |
| `0x812` | one-by-one input upsample fast-path flag |
| `0x81d` | resize and texture-engine-present flag |
| `0x3f8`, `0x400`, `0x410` | patch-width clamps for max-pool, floor, and non-pool |
| `0x228`, `0x238`, `0x240`, `0x248` | cost-model frequency, cycles, and core count |

Table: The fields of the 0x348-byte hardware-abstraction parameter blob and the route each drives. {#tbl:c35-halmap}

The concrete numeric values at these offsets are in a constant data section behind a packed selector the target constructors pass: `H13` and `H17s` pass the identical selector, so both share a base hardware-abstraction descriptor and diverge only in the literals.
The per-target literals are deferred; the offsets, consumers, and mechanism are pinned here.

## Per-family cost model

`getDeviceInfo` fills a device-information struct from the family enum and the operation size.
It writes a constant `0.0008` at offset `0x30`, a size-scaled pair at offsets `0x14` and `0x18`, a bandwidth clamp at offset `0x20`, and a setup-latency and throughput pair at offsets `0x8` and `0xc` from a data block.
The dispatch reads family 5 and 6 as the A16 and A17 tier, family 4 as the M1, and family 3 as A12, and buckets the size at thresholds of 6, 10, 20, and 40.
[Table](#tbl:c35-costmodel) gives the bandwidth-clamp sequence and setup ramp per family in the analytic cost model.

| Family | Bandwidth clamp sequence by size bucket | Setup ramp |
| --- | --- | --- |
| A12 | 50, 100, 200, 400, 800 | 1.84 to 14.72 |
| A13, M1 | 50, 100, 200, 400, 800 | 1.84 to 7.36 |
| A16, A17, M5 | 62.5, 125, 250, 500 | 1.84 to 7.36 |
| default (A11, A14, A15) | 34.1, 68.2, and on | 1.84 to 13.57 |

Table: The bandwidth-clamp sequence and setup ramp per family in the analytic cost model. {#tbl:c35-costmodel}

The M5 is modeled with a higher per-bucket throughput but a clamp ceiling of 500 against the M1 and A12 ceiling of 800.

## A15 codegen branch

The A15 family is its own code-generation tier, not a relabel of A14, and the difference is visible in three independent locations in the compiler.
The following is decode-derived from the static compiler image and is not measured on A15 silicon.

`getDeviceInfo` has a dedicated `anec::A15` branch.
The dispatch range-tests the family value and special-cases `A14` and `A15` separately, so the `A15` arm fills the device-information struct from its own cost-table block rather than falling to the A14 arm or the shared default.
A15 thus has distinct cost-model constants, which is the data-side proof that the family is a distinct capability tier.

The operation legality matches.
Forty-five `MinimumFamily<N>` template instantiations have the F4 floor, the A15-and-above floor, against twenty-nine at the A14 floor, so A15 adds a concrete operation set over A14 rather than reusing the A14 set.
The distinct compute operations at that floor are `sin`, `cos`, and the global arg-reductions.

Beyond the H15 constructors in the chapter 34 census, the compiler carries recognized target strings for a wider H15 die family, `h15g`, `h15m`, `h15p`, `h15s`, `h15c`, and `h15d`.
The `h15m` suffix is an extra M-class target that the h13 and h14 families do not list.
A15 is also the only tier whose per-chip interchange-format table accepts YUV420 image input, a 4CC route that the surrounding families drop, which is consistent with A15 being a distinct capability tier rather than a renamed A14.
The A15 cost-table constants and the YUV420 acceptance are decoded from the compiler; the numeric behavior of the A15 operation set, such as any accumulator or rounding change against A13 and A14, is an A15-silicon measurement that this guide does not have.

## Named validity gate

Only the M1 family has a named per-family validity gate.
`IsValidForH13` is the gather gate: it asserts the gather-axes size is 3, the data batch and depth are 1, the index channel is 3 with width and depth 1, plus an axis-pattern check.
`IsValidForH13GatherNCH` holds the channel-height-width-layout gather restriction.
No equivalent named gate exists for any other family; the rest gate numerically through the hardware-abstraction clamps, and the compression gate `IsValidForCompression` is hardware-abstraction-driven with the universal rule that only kernels of at least eight bits are compressible.

## Cross-compiled across the family

The per-generation versioning that chapter 23 reads out of the descriptor accessors can be observed directly in the emitted program.
Because the compiler is one binary that lowers for any target from any machine, the target is a command-line flag rather than a property of the host: a single `ANECompile` build, here `9.509.0`, lowers the same network for `h13` through `h17` while only the `-t` target varies.
A diff across those programs isolates the per-generation divergence from the compiler version and the operating-system version, which a diff across physical machines confounds.
Lowering one convolution and one matmul for the five base M-series targets, `h13` through `h17`, gives the comparison that [Table](#tbl:c35-crossgen) records.

| Target | Silicon | cpusubtype `+0x08` | Convolution op-config word | Aperture base-record stride |
| --- | --- | --- | --- | --- |
| `h13` | A13, M1 base | `0x04` | `0x5042a063` | 8 bytes |
| `h14` | A14, M2 base | `0x05` | `0x5042a0c3` | 16 bytes |
| `h15` | A15, M3 base | `0x06` | `0x5042a0c3` | 16 bytes |
| `h16` | A16, M4 base | `0x07` | `0x5042a0c3` | 16 bytes |
| `h17` | A17, M5 base | `0x09` | `0x5042a0c3` | 16 bytes |

Table: One convolution and one matmul cross-compiled for the five base M-series targets from a single machine and compiler, with the per-generation fields read from the emitted container. {#tbl:c35-crossgen}

The cpusubtype at header offset `0x08`, the codegen revision chapter 23 reads as `0x4` on the M1, steps once per generation across the family: `0x04`, `0x05`, `0x06`, `0x07`, and `0x09`, the M5 base skipping `0x08`, with the A19 target `h18` continuing to `0x0a`.
The convolution op-config word that chapter 23 decodes as the version-7 value `0x5042a063` holds on the M1 and becomes `0x5042a0c3` on the M2 and every later base target.
The high half-word `0x5042` is stable, as chapter 23 states; it is the low byte that carries a generation component, `0x63` on the version-7 image and `0xc3` from the A14 image on.
The version-7 op-config words in the `0x5000a0..` range, `0x5000a021` and `0x5000a421`, appear only in the M1 image and are absent from the M2 image onward, the byte-level form of the version bump chapter 23 describes.

What does not move is the addressing aperture and the descriptor prologue.
The aperture virtual addresses, input `0x30004000` and the `0x30008000`, `0x3000c000`, and `0x30010000` that follow, are byte-identical across all five targets, as is the descriptor header through its first `0x300` bytes.
What moves is the register program below them: the relocation records that carry the aperture base addresses are packed at an 8-byte stride on the M1 and move to a 16-byte stride from the M2 on, the concrete instance of the rule that the byte offsets and field widths move between versions (chapter 23) and of the per-generation descriptor-offset [Table](#tbl:c35-descoffset).
Both operations show the same stride shift at the same boundary.

The largest single step is the M1 to the M2, the version-7 descriptor to the A14 descriptor.
The op-config word, the relocation stride, and the version-7 config words all change there, while the M2 through the M5 share the op-config word and the stride and differ mainly in the cpusubtype stamp and the kernel-section size, the convolution image growing from 64 KB on the M1 through the M3 to 96 KB on the M4 and M5.
This is the program-format counterpart of chapter 34's observation that the operation surface stops expanding at A15 while the upper tiers add cores: the format is largely settled after A14, and the later changes are incremental.
This comparison is the compiler's emitted program for each target, read from the cross-compiled container, not an on-silicon run on the M2, M3, and M4, which this guide does not have; it confirms the decompile-derived per-generation versioning of chapter 23 against the actual byte stream, with the M1 and M5 endpoints also confirmed on silicon.

## Apple documents

Per-family code generation has no public counterpart, and this chapter reports it as reverse-engineered across the target set.
The public conversion tools document the frontend operation set and the palettization, quantization, and pruning passes that produce a model, and the compute-unit selector that requests the engine [AppleCoreMLTools].
They do not document the family enum, minimum-family operation trait, per-chip hardware-abstraction parameters, or route and format selection that makes one slice saturate on the M1 and run clean on the M5.
