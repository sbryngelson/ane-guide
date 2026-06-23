# 36. Predicted upper tier

> Two datapaths are above the M5 in the compiler that no part measured here can run: an fp8 weight and activation format, and a multi-die collective communication layer.
> Both are present as decoded structure, gated off on every family this guide reached, and stated as predicted, not measured.
> The fp8 gate is the capability byte at offset `0x52d`, set on H18 alone; the collective-enable byte at offset `0x48b` reads zero on all 28 targets.

## 64-core ceiling

The largest die this guide measured is the 16-core H17s, and the line ends at the 64-core H17d that no part here could run.
The decode shows that nothing extra is family-gated for the 64-core case beyond the count itself, and the following is read from the compiler rather than measured on a 64-core part.
Core count is a runtime parameter, the `ne_core_count` value sourced from the hardware-abstraction blob, not a hard-coded per-family constant.
`getDeviceInfo` takes that count as its second argument and buckets it against the thresholds 7, 10, 20, and 40, which straddle the four die sizes 8, 16, 32, and 64 held by the `g`, `s`, `c`, and `d` target suffixes.
Several device-information fields then scale linearly with the count, and the top bucket reached on the A14-and-above branch stores the 800-unit peak.
The compiler thus emits a 64-core-scaled cost model for H17d purely from the count of 64, and the geometry is decoded while the realized 64-core throughput stays an on-silicon measurement.

The two datapaths above the M5 are the fp8 weight format and the multi-die collective layer, which [table](#tbl:c36-overview) gives field by field, with each gate and its state on the M5.

| field | E4M3 | E5M2 | collective layer |
| --- | --- | --- | --- |
| sign bits | 1 | 1 | not applicable |
| exponent bits | 4 | 5 | not applicable |
| mantissa bits | 3 | 2 | not applicable |
| exponent bias | 7 | 15 | not applicable |
| max finite magnitude | 448 | 57344 | not applicable |
| role | gated weight and activation | fp16 conversion | mesh reduce, gather, slice |
| family gate | `0x52d` set on H18 only | folds to fp16 broadly | `0x48b` zero on all 28 |
| register encoding | present from A17 encoder | output slot on modern encoders | every setter a stub |
| state on M5 | off | conversion available | inert |

Table: The decoded fields of the two fp8 formats and the collective layer, with their gates and their state on the M5. {#tbl:c36-overview}

## fp8 datapath

The compiler has two eight-bit floating-point formats, and they are not symmetric in role.
They are in the internal `ZinTensorFormat` code-generation and direct-memory-access enumeration, not in the serialization data-type enumeration that the operation attributes use.
The serialization enumeration runs codes 0 through 10 with fp16 at 3, fp32 at 4, and int8 at 2, and has no fp8 codepoint at all; the fp8 formats are recovered from the byte-size map `_ZinTensorFormatGetSizeInBytes`.
[Table](#tbl:c36-tensorfmt) gives the fp8 codepoints in the internal tensor-format enumeration, with the integer and half-precision neighbors that bound them.

| `ZinTensorFormat` code | Format | Bytes |
| --- | --- | --- |
| 1 | int8 | 1 |
| 2 | uint8 | 1 |
| 3 | fp16 | 2 |
| `0xb` (11) | fp32 | 4 |
| `0xc` (12) | E4M3 | 1 |
| `0xd` (13) | E5M2 | 1 |
| `0xe` (14) | two-byte live-input format | 2 |
| `0x10` (16) | int4 | under 1 |

Table: The fp8 codepoints in the internal tensor-format enumeration, with the integer and half-precision neighbors that bound them. {#tbl:c36-tensorfmt}

The compiler has the full set of fp8 variant types as C++ symbols, `e4m3_t` with 1273 references and `e5m2_t` with 176, alongside the finite-and-NaN and unsigned-zero variant type classes.
The shipping format is the E4M3FN variant, proven by the $\pm 448$ saturation bound that `ZinPadLayer::ValidateBackgroundPaddingValue` enforces with the string "in [%u, %u] for e4m3 format" and the NaN-only special case in `ZinE4M3Expand`.

E4M3 is the gated weight and activation format.
A value decodes from a sign bit, four-bit exponent, and three-bit mantissa as

$$x = (-1)^s \, 2^{e - 7} \left(1 + \frac{m}{2^3}\right)$$

for a normal value, with an exponent bias of $7$ and a maximum finite magnitude of $448$.
The shipping variant is E4M3FN, the finite-and-NaN form: it has no infinity, the all-ones exponent with a full mantissa encodes NaN, and a value past $\pm 448$ either maps to NaN or clamps to the maximum normal.
A one-bit mode held per direct-memory-access source controls the overflow behavior, $0$ for NaN and $1$ for saturate, so a narrowing cast picks the bit pattern $\mathtt{0x7e}$ for the saturated maximum or $\mathtt{0x7f}$ for NaN.
The narrower exponent and the $\pm 448$ ceiling cap the representable range below half precision.

E5M2 is a conversion format, not a gated weight format.
It decodes as

$$x = (-1)^s \, 2^{e - 15} \left(1 + \frac{m}{2^2}\right)$$

with five exponent bits, a two-bit mantissa, and the same exponent bias of $15$ as half precision.
That shared exponent is why E5M2 is the high byte of an fp16 value: the conversion to fp16 is a left shift of eight bits, and the inverse is the top byte with round-to-nearest and an infinity clamp.
E5M2 folds to fp16 by a direct-memory-access conversion, so it is broadly available wherever that conversion path exists, and the upscaler layer accepts it as input.
The fold asymmetry is exact in the conversion functions.
`ZinE5M2ToF16` is a left shift of eight bits, the literal high byte, and `ZinF16ToE5M2` is the top byte with round-to-nearest and an infinity clamp at `0x7c00`.
E4M3 is not so trivial.
`ZinE4M3Expand` is a bit-surgery decode with sign at bit 7, the four-bit exponent at bits 6 through 3, and the three-bit mantissa at bits 2 through 0, handling the subnormal and NaN encodings.
`ZinF32ToE4M3` narrows and selects the overflow byte, `0x7e` for the saturated maximum and `0x7f` for NaN.
The fold predicate `IsFormatDMAConvertibleToFP16` returns `(fmt < 0xe) & (0x2ff0 >> fmt)`, and the mask `0x2ff0` includes code 13 and excludes code 12.
E5M2 thus folds by direct-memory-access conversion and E4M3 cannot, which is the binary-level reason E4M3 requires the native datapath that is gated.

The family gate is a single hardware-abstraction-layer capability byte at offset `0x52d`, the E4M3 direct-memory-access and kernel-format capability.
It is clear on the M1 generation, clear on the M5 generation, and clear on the intermediate families, and it reads set on the H18 family alone of the twenty-eight compiler targets.
The master per-format direct-memory-access validity function `CheckValidDMAFormat` takes the four capability bytes `HAL[0x52c]`, `HAL[0x52d]`, `HAL[0x52e]`, and `HAL[0x685]`.
It validates fp32 against `0x52c`, E4M3 against `0x52d`, E5M2 against `0x52e`, and the two-byte live formats against `0x685`, with every code below `0xb` always valid.
Three operation-semantics sites re-check the byte and abort with "E4M3 is not supported on this architecture": the dequantize validator at line 129078, the quantize validator at line 227466, and a third semantics validator at line 3407368.
Below that runtime gate is a harder compile-time limit on the M1 generation.
The M1 task-descriptor encoder packs the source element format into a two-bit field, which holds only the integer and half-precision codes and has no bit space for the E4M3 codepoint.
Feeding fp8 to the M1 encoder thus aborts the compile rather than refusing the operation at runtime.
The encoder that does have the E4M3 codepoint widens that field to three bits and adds a distinct output slot, and that wider encoder is present from the A17 generation onward even though the runtime capability at `0x52d` fires only on H18.

The operation-side gate matches the encoder, decoded from the compiler and not measured on an H18 part.
The `MinimumFamily<N>` trait that chapter 35 reads as the operation-legality floor has a high-N set, a few operations at N=5, N=6, and N=7, the A16, A17, and A18 floors, and the fp8-bearing operations are in that set.
The fp8 converters and the quant-unit storage are compiled into the one byte-identical image, so an fp8 operation parses and type-checks on any target.
Native execution depends on both that high-N family floor and the `0x52d` capability byte, which is why the datapath is present yet inert below H18.

The format-register delta between the M1 and the H18 encoder is exact.
The M1 generation builds the generation-5 and generation-7 descriptors, whose `SetCommonInFmt` writes a two-bit field at `this+0x48` holding only int8, uint8, and fp16, and the code `0xc` aborts with "Error: Invalid Common InFmt E4M3".
The generation-20 descriptor that the A17 generation builds packs all three format fields into a 32-bit word at `this+0x228`, widening each lane to three bits.
`SetCommonInFmt` gives E4M3 the source-one code 4 at bits 2 through 0, `SetCommonSrc2InFmt` gives it the source-two code `0x20` at bits 5 through 3, and `SetCommonOutFmt` gives it a distinct output slot `0x100` at bits 8 through 6.
[Table](#tbl:c36-fmtreg) gives the task-descriptor format-register delta, showing that the M1 encoder has no bit space for an E4M3 codepoint while the generation-20 encoder widens each lane to three bits.

| Field | M1 generation-5 at `this+0x48` | H18 generation-20 at `this+0x228` |
| --- | --- | --- |
| source-one format | two-bit lane, int8, uint8, fp16 only | three-bit lane, plus E4M3 as 4 |
| source-two format | folded into the two-bit space, no distinct E4M3 code | three-bit lane, E4M3 as `0x20` |
| output format | two-bit lane, E5M2 reuses the fp16 slot `0x20` | three-bit lane, E4M3 as `0x100` |
| E4M3 codepoint | absent, a compile-time assert | present as 4, `0x20`, `0x100` |
| E4M3 overflow register | stub assert on the generation-8 setter | `this+0x2fc` and `this+0x300`, bit 24 |

Table: The task-descriptor format-register delta across the upper-tier generations. {#tbl:c36-fmtreg}

The overflow mode is a per-source register field on the generation-20 encoder, `SetTileDmaSrc1E4M3Overflow` writing bit 24 of `this+0x2fc` and `SetTileDmaSrc2E4M3Overflow` writing bit 24 of `this+0x300`, where 1 selects saturate and 0 selects NaN.
On the older encoders the same setter is a stub that asserts "E4M3Overflow is not supported" only when the overflow option is engaged.

In the multiply array fp8 is an input width, not an accumulator width.
An E4M3 weight expands to fp16 going into the multiply, the products accumulate in the same wide register every family uses, and the output port rounds the result to fp16.
No fp8 accumulator type, register field, or symbol exists in the compiler: the generation-20 format word encodes only the source-one, source-two, and output element formats, with no accumulator-format field.
E4M3 is a native kernel format, code 6 in the kernel-format set that `ZinSetFormat` admits through the mask `0x3b`, alongside int8, uint8, fp16, and fp32; E5M2 is absent from that set, so it is a conversion and upscaler-input format rather than a kernel format.
Throughput runs on the same double-rate path int8 uses, gated by `ZinDoubleMacMode::CanUseDoubleMacModeBasedOnFormats`.
A one-byte activation against a one-byte kernel of the same numeric class is eligible for the double-multiply mode, the eligibility being the exclusive-or term that is true only when the activation-float bit equals the kernel-float bit.
An E4M3 activation against an E4M3 kernel is both one-byte and both float, so it runs at twice the per-element rate into the fp16 accumulator, while a mixed int8-against-E4M3 pair is not eligible.
E4M3 is symmetric-only, with the zero point forced to zero, the same constraint the M1 imposes on int8.
The quantize and dequantize validators reject a zero point with "Zero point is not supported for quant with E4M3 output format", and the same rule reaches the palette layer.
E4M3 weights stream as a palette through `ZinIrWeight::DePalettizeWeightData<e4m3_t>`, a codebook of E4M3 values indexed by a packed stream at bit-widths 1, 2, 3, 4, 6, and 8.
This is the identical machinery the int4, int8, and fp16 palettes use under template specialization, dequantized as scale times E4M3 with no zero point.

## Multi-die collective layer

[Table](#tbl:c36-collops) gives the collective operations of the multi-die dialect, their backend operation classes, and the unit-type code of each.

| silc mnemonic | C++ operation class | role | unit-type code |
| --- | --- | --- | --- |
| `silc.all_reduce` | `SilcAllReduceOp` | reduce a tensor in place across a mesh axis | 78 |
| `silc.all_gather` | `SilcAllGatherOp` | concatenate shards into a replicated tensor | 76 |
| `silc.all_slice` | `SilcAllSliceOp` | scatter a replicated tensor into shards | 75 |
| `silc.mesh` | `SilcMeshOp` | declare the device mesh | none |
| `silc.call` | `SilcCallOp` | per-die single-program call | 79 |

Table: The collective operations of the multi-die dialect, their backend operation classes, and the unit-type code of each. {#tbl:c36-collops}

The collective layer is a cross-die data path, not the independent per-job steering a multi-die part otherwise does.
The kernel load-balancer steers whole independent submissions to the least-busy engine die and never exchanges tensor data between them, and the driver supports up to four dies.
The M1 and the M1 Max each register a single engine die, so this steering engages only on a multi-die part such as the Ultra.
The compiler's collective instead splits one tensor across dies and exchanges partials, threaded through an `optional<ZinIrDeviceMesh>` that `GetAndValidateSpmdDeviceMesh` and `ZinParseDeviceMeshAttributes` build, so a program compiled with a device-mesh attribute gets a real all-reduce, all-gather, or all-slice across the mesh rather than per-die placement.
The cross-die move itself is the `HandleCcdmaLayer` direct-memory-access primitive below, and `ValidateMeshAxesInTensorFamily` is the family gate on whether a given die admits the mesh.
This contrast is decode-derived; whether a two-die Ultra in any reached family accepts the mesh path is a multi-die measurement this guide does not have.

It is a distinct intermediate-language dialect, `mlir::silc`, whose closed operation set is the three collectives above over a device mesh, plus the mesh declaration and the per-die call.
The operation-class list is exactly these, with no reduce-scatter and no broadcast, the all-slice operation standing in for the scatter.
The collective operations have the attributes `mesh`, `mesh_axes`, `sharding`, the `members` membership list, and `reduce_op`.

The reduction kind is an enumerated attribute decoded directly from the packed-string compare in `symbolizeReductionKind`, which [table](#tbl:c36-redkind) gives token by token.

| Token | Integer |
| --- | --- |
| `sum` | 1 |
| `max` | 2 |
| `min` | 3 |
| `product` | 4 |
| `mean` | 5 |
| miss | 0, invalid |

Table: The reduction-kind enumeration the all-reduce reduce-operation attribute holds. {#tbl:c36-redkind}

The mesh is an N-dimensional grid of dies by engines-per-die, a vector of per-axis extents held in `ZinIrDeviceMesh`, which exposes the die count, total engine count, and engines-per-die count.
A device identifier maps to a mesh coordinate and an engine index by a mixed-radix split at the die axis, which `ZinSPMDUtils::AneIndexFromDeviceId` computes:

$$\mathrm{ane} = \sum_{i \ge d} \mathrm{id}[i] \prod_{j > i,\, j \ge d} \mathrm{ext}[j] \; + \; \left( \sum_{i < d} \mathrm{id}[i] \prod_{j < i} \mathrm{ext}[j] \right) \cdot \mathrm{anesPerDie}$$

where $d$ is the die-axis split index: the axes above the die axis fold into the within-die engine offset, and the axes below fold into the die index, multiplied by engines-per-die.
The inverse decode is `DeviceIdFromAneIndex`.
A layer runs on a die through `ZinEngineLayer::RunsOnDeviceId`.
A layer not assigned to the single-program path runs on all dies.
Otherwise it emits on a die only when the engine index for that die is a key in the layer's engine-set map, which is how the `members` attribute becomes a per-die emit decision.

A sharding attribute maps each tensor axis to a mesh axis, splitting that axis into one shard per device along the axis.
The split requires even divisibility, which the strings "Input tensor dimension must be divisible by the number of shards along the tensor dimension" and "Kernel dimension must be divisible by number of shards" enforce.
The compiler rejects sharding the same mesh dimension twice and allows a replicated section only on a multi-die network.

The reduce-across-the-mesh operation lowers to a collective direct-memory-access that holds a hardware atomic read-modify-write: the reduction happens in memory as the direct-memory-access writes into a shared buffer, which is why the operation requires in-memory-reduction support that the single-die families lack.
The reduction-to-atomic map is decoded from the `ZinIrReductionTypeToZinAtomicOpType` switch, which [table](#tbl:c36-atomic) gives along with the reductions that have no hardware atomic and are rejected.

| Reduction type | Atomic-op register value |
| --- | --- |
| 1 | 4 |
| 2 | 3 |
| 4 | 1 |
| 5 | 2 |
| 8 | 5 |
| 9 | 6 |
| 10 | 7 |
| 3, 6, 7, 11 | assert, no hardware atomic |
| other | 0 |

Table: The reduction-type-to-atomic-operation register map for the all-reduce collective, with the reductions that have no hardware atomic and are rejected. {#tbl:c36-atomic}

The atomic configuration packs as `(atomicOp & 0xff) | (atomicDataType << 8)`, and the input supplied to the collective must arrive through a bypass pass-through, asserted by "Input to inter-die AllReduce should be NEBypass".
The all-reduce lowering itself further asserts "AllReduce is currently not supported for architectures that do not support in-memory reductions".

`HandleCcdmaLayer<Nu>` programs the collective direct-memory-access into the task descriptor, instantiated for the generations $\{1, 4, 5, 6, 7, 8, 10, 11, 17, 19, 20\}$.
Each opens with the collective-enable gate and the per-die predicate before driving the setters in a fixed order.
That order is the source mode, the counter mode, the data size, five shape words, four destination strides, the destination base address, the optional constant, four source strides, the source base address, the wait-event address from the mesh symbol, the counter address, the atomic data type, the atomic operation, the counter amount, and the wait-event value.
The base addresses are all emitted through `ZinSPMDUtils::GetSymbolOffsetToBaseAddr`, the extended-addressing path that needs the cross-die address-reach byte.

The layer is decoded but inert in the compiler this guide reads.
Every collective direct-memory-access register setter is a stub that asserts "CCDMA is not supported for this arch", in every task-descriptor generation present including generation 20, the would-be Ultra path: the body calls the full setter sequence, but the setters abort.
The collective-enable capability byte at offset `0x48b` reads zero on all twenty-eight targets including the M-class and Ultra-reference targets.
[Table](#tbl:c36-capbytes) gives the collective and cross-die capability bytes decoded across the twenty-eight targets, with the collective-enable byte clear everywhere.

| Byte | Capability | Floor | M1 (H13) | A14 | A15 | M5 (H17s) | H18 | M-class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0x48b` | collective-enable | none in this build | 0 | 0 | 0 | 0 | 0 | 0 |
| `0x55c` | cross-die extended addressing | A14 | 0 | 1 | 1 | 1 | 1 | mixed, M11 and U2, U3 set, U1 clear |
| `0x564` | multi-die hazard tracking | A14 | 0 | 1 | 1 | 1 | 0 | 0 |
| `0x687` | multi-die remote dependency | A15 | 0 | 0 | 1 | 1 | 0 | 0 |
| `0x4a4` | M-class secondary cap | M-class only | 0 | 0 | 0 | 0 | 0 | 1 |

Table: The collective and cross-die capability bytes decoded across the twenty-eight targets, with the collective-enable byte clear everywhere. {#tbl:c36-capbytes}

The compiler thus has the front end of the collective in full: the operation set, mesh and sharding math, reduction-to-atomic map, and per-die dispatch.
No family in this compiler arms the register encoding that would drive the engine.

A second-layer gate is in the source-direct-memory-access emitters, where the single-program flag drives an assert "This target does not allow sharding or SPMD functions".
A third gate in the tasklet emitter asserts "No tasklet for given architecture" when the per-section tasklet bit and the single-program predicate are not both set.
On the M1 all three gates fail, so the M1 is single-die.

The pieces the single-die M1 silicon does touch are the on-die ordering bits the same machinery shares.
Those bits are the layer-two barrier, event masks, and remote-dependency bookkeeping, none of which is the cross-die reduction itself.
On the M1 generation-10 descriptor the layer-two barrier sets bit 23 of `TD+0x134` and the forward barrier sets bit 30.
The event setter writes a 26-bit signal mask into `TD+0x10` and a 26-bit wait mask into `TD+0x18`, and a direct-memory-backed event is rejected with "DRAM Events not supported for architecture".
The distributed unit is the tasklet, the multi-die variant of the descriptor instruction, one tasklet per participating die.

## What requires newer parts

The fp8 datapath requires an H18-class part to set the capability byte and exercise the native E4M3 multiply.
The collective layer requires a multi-die part and a newer compiler that arms the collective-enable byte and replaces the register stubs with a real encoding.
Until that hardware and that compiler are in hand, the encodings here stand as the predicted upper tier of the line.
