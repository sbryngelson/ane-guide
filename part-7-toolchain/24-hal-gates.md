# 24. HAL and capability gates

> The compiler is one binary that builds any chip in the line from a per-chip data table, the hardware abstraction layer, read at compile time.
> The table holds scalar fields indexed by byte offset for the numeric limits and a dense capability-byte region at offsets `0x48f` through `0x8cc`, each byte read as `hal[offset] & 1` to gate one operation or format.
> Operation legality is declared on the operation as a `MinimumFamily<N>` trait: native only when the target family index is `N` or greater, and decomposed below the floor, with no compute operation floored above A15.
> A capability in the table attests support at the layer that reads it and does not prove the operation runs: three-dimensional convolution has its kernel-depth attestation at `0x70` and fails backend lowering on every device mask.

The compiler that targets the Apple Neural Engine is one binary that builds any chip in the line on demand.
What separates one target from the next is a per-chip data table, read at compile time, that records every size limit and every per-operation switch for that silicon.

## HAL property table

The hardware-abstraction-layer table is the compiler's profile of a target, one packed structure that holds both the numeric limits and the feature gates for a chip.
[Table](#tbl:c24-hal-scalars) gives representative scalar fields of the hardware-abstraction table, each with its offset, meaning, M1 value, and the generation at which it changes.

| Offset | Field | Meaning | M1 (H13) | Changes at |
| --- | --- | --- | --- | --- |
| `0x1b8` | `max_operand_bytes` | on-chip SRAM working set | 2 MB | constant across the line (1 MB on M9) |
| `0x1c0` | `dram_alignment` | DMA width granule in bytes | 16 | constant (1 only on the small profiles) |
| `0x1c8` | `l2_bank_align` | DMA bank-conflict modulo | 64 | constant |
| `0x1f0` | L2-resident buffer threshold | dedicated-buffer trip | 0 | 32768 at A15, 262144 at A16 |
| `0x200` | dense kernel-memory cap | non-streamed weight ceiling | 64 KB | the fold-path budget |
| `0x210` | streamed kernel-memory cap | streamed weight ceiling | 16 MB | the stream-path budget |
| `0x218` | instruction or segment alignment | record packing granule | 256 | 16 at A14 |
| `0x228` | `ne_perf_cycle_divisor` | cost-model per-cycle divisor | 64 | 32 on H11, 16 on M9 |
| `0x238` | `num_nes` | NE-core count | 4 (base) | die-keyed: 4, 8, 16, 32, 64 |
| `0x288` | extended dual-kernel-memory mode | 16 MB versus 64 KB select | 0 | 0 on all 28 targets |
| `0x70` | `max_large_conv_kernel_dim_z` | 3D-conv kernel depth | 16 | capability attested at A13 (1 below) |
| `0x138` | `max_tensor_width` | maximum tensor width | 16384 | 65536 at A16 |
| `0x158` | `max_tensor_depth` | maximum tensor depth | 16384 | 1 below A13, 65536 at A16 |
| `0x3f0` | reduction-via-transpose extent | reduction route threshold | 192 | 384 at A15 |
| `0x400` | `pe_min_patch_width_log2` | the $2^4 = 16$-pixel tiling floor | 4 | constant M1 and M5 |
| `0x580` | cost-model policy name | roofline anchor string | `Simple` | `None` on older and small profiles |
| `0x668` | interchange-format map size | count of accepted image formats | 3 | 13 at A14, 16 at A15, 14 at A16 |

Table: Representative scalar fields of the hardware-abstraction table; the full decoded register map is in Appendix C. {#tbl:c24-hal-scalars}

The hardware abstraction layer is a single packed structure the compiler constructs for its target, holding two kinds of entry.
The first is a block of scalar fields, indexed by byte offset, that record numeric limits: maximum kernel sizes, maximum tensor dimensions per axis, the on-chip working-set size, data-movement alignment granule, and cost-model curve.
The second is a dense region of single-byte boolean flags, the capability bytes, each gating one operation or one format on or off for the target.
A family of constructors, one per architecture, builds the structure per target inside the compiler.
The scalar region runs from offset `0x18` to roughly `0x348` as plain data, with non-scalar members such as the format map and the cost-model curve extending past it.
The capability-byte region occupies offsets `0x48f` through `0x8cc` and holds on the order of 165 single-byte flags, of which 24 have recovered field names and the rest are enumerated by offset and classified by the family that enables them.
The compiler reads a scalar as a value at its offset and a capability byte as `hal[offset] & 1`, so every limit and every gate is one indexed read into this one table.
The same structure holds the cost model: a policy-name string at `0x580`, frequency-to-efficiency curve at `0x7a8`, and per-cycle divisor at `0x228`, which the roofline of chapter 18 reads from this table rather than from a separate file.

[Listing](#lst:c24-hal-struct) gives a partial C view of the structure, with each selected scalar field and the capability-byte region at its recovered offset.

```c
/* ZinIrHalParameters, selected fields at their byte offsets (M1/H13 values) */
struct ZinIrHalParameters {
    /* ... */
    uint64_t max_large_conv_kernel_dim_z;  /* 0x70:  3D-conv kernel depth   = 16  */
    /* ... */
    uint64_t max_tensor_width;             /* 0x138: max tensor width       = 16384 */
    /* ... */
    uint64_t max_operand_bytes;            /* 0x1b8: SRAM working set       = 2 MB  */
    uint64_t dram_alignment;               /* 0x1c0: DMA width granule      = 16    */
    uint64_t l2_bank_align;                /* 0x1c8: DMA/L2 bank count      = 64    */
    /* ... */
    uint64_t num_nes;                      /* 0x238: NE-core count          = 4     */
    /* ... */
    uint8_t  cap_bytes[0x8cd - 0x48f];     /* 0x48f..0x8cc: per-op capability flags */
};
```

Listing: A partial C view of the per-target hardware-abstraction structure, with selected scalar fields and the capability-byte region at their recovered offsets. {#lst:c24-hal-struct}

The capability bytes are read one at a time as `hal[offset] & 1`, for example the texture engine at `0x81d` and the kernel-streaming master at `0x48f`.

## Operation gate

Operation legality is declared not in the HAL table but on the operation itself, as a trait the compiler attaches to every backend operation.
The trait is a minimum-family index: the operation `MinimumFamily<N>` is natively legal only inside a compilation whose family index is `N` or greater, and below that floor the compiler decomposes it into legal operations.
The family index orders the generations: `A11Legacy` is 0, `A12` is 1, `A13` is 2, `A14` is 3, `A15` is 4, and so on, with the M1 at `A13` and the M5 at `A17`.

The check the compiler runs on each backend operation is the trait floor against the target family index, which [listing](#lst:c24-family-gate) gives as the native-or-decompose decision.

```python
# mlir::OpTrait::anec::MinimumFamily<N>: native iff target family >= N
def op_is_native(op, target_family):
    return target_family >= op.minimum_family   # e.g. softmax N=2 (A13), sin N=4 (A15)

def lower_op(op, target_family):
    if op_is_native(op, target_family):
        emit_native(op)                          # one anec op
    else:
        decompose(op)                            # rewrite into ops legal below the floor
```

Listing: The minimum-family gate, where an operation is emitted natively when the target family meets its floor and decomposed otherwise. {#lst:c24-family-gate}

The M1 has family index two and the M5 has family index six, so an operation with floor four, such as sin, is native on the M5 and decomposed on the M1.

The floors fall into a small number of tiers.
The base tier, family 0, holds the operations every engine runs: convolution, matrix multiply, pooling, the elementwise and activation set, reshape, transpose, and concat.
At A13 come softmax, the normalizations, the reductions, fused attention, and the square-root and error functions.
A14 brings the texture-engine samplers, crop-resize, and resample; A15 brings native sin and cos.
No compute operation floors above A15, so the newest generations add core count and clock rather than new operations.

The two gate mechanisms work together.
A capability byte read as `hal[offset] & 1` decides a route inside a single operation, for example whether the texture engine at byte `0x81d` is present, which on the M1 reads 0 and forces resize to a decomposition.
The minimum-family trait decides whether the operation is native at all.
When either gate is closed, the compiler either emits a decomposition into legal operations or rejects the operation with a message naming the architecture, depending on whether a legal decomposition exists.

[Table](#tbl:c24-floors) gives the minimum-family floors a developer reaches, each with the families it is native on and its representative operations.

| Floor | Native on | Representative operations |
| --- | --- | --- |
| F0 | all families | convolution, matmul, pooling, elementwise, reshape, transpose, concat |
| F2 (A13+) | A13 onward | softmax, layer and instance and batch norm, reductions, attention, erf, sqrt |
| F3 (A14+) | A14 onward | crop-resize, resample |
| F4 (A15+) | A15 onward | sin, cos, global argmin and argmax |

Table: The minimum-family floors a developer reaches, with the families each is native on and representative operations. {#tbl:c24-floors}

Because the floor is an attribute of the operation and the limits are a table keyed to the chip, the per-chip difference is data, not code.
The compiler text that rewrites an operation is identical across the family, and the chip selects a different limit, gate, or decomposition strategy from the table beneath it.

## Capability-byte gates across the line

A capability byte is a single-byte switch in that table that turns one operation or feature on or off for a target.
[Table](#tbl:c24-capbytes) gives the named capability bytes, each with its gate and its value across the M1 and the later generations.

| Byte | Gate | M1 (H13) | A14 | A15 | A16 | A18 |
| --- | --- | --- | --- | --- | --- | --- |
| `0x48f` | kernel-streaming master, the 64 KB to 16 MB select | 1 | 1 | 1 | 1 | 1 |
| `0x494` | square-after-reduction fusion | 0 | 1 | 1 | 1 | 1 |
| `0x4a9` | dropout and random | 0 | 0 | 1 | 1 | 1 |
| `0x4f2` | global argmin and argmax | 1 | 1 | 1 | 1 | 1 |
| `0x529` | per-format kernel-stride enable, the palette stream | 1 | 1 | 1 | 1 | 1 |
| `0x52d` | fp8 E4M3 kernel format | 0 | 0 | 0 | 0 | 1 |
| `0x563` | FIFO-mode direct memory access | 0 | 0 | 0 | 0 | 1 |
| `0x815` | softmax, native | 1 | 1 | 1 | 1 | 1 |
| `0x816` | instance normalization, native | 1 | 1 | 1 | 1 | 1 |
| `0x81a` | local-response normalization, native | 1 | 1 | 1 | 1 | 1 |
| `0x81d` | texture engine | 0 | 1 | 1 | 1 | 1 |

Table: The named capability bytes. {#tbl:c24-capbytes}

The texture engine at byte `0x81d` is the largest M1 functional gap: it reads 0 on the M1 and 1 from A14 onward.
It gates resize, crop-resize, resample, affine transform, hardware gather, and symmetric padding all together, so each of those routes through a software decomposition on the M1.
The fp8 byte `0x52d` is set on the A18 generation alone of the 28 targets, so the M5, an A17 part, does not have it.
The streaming master at byte `0x48f` and the palette-stream byte `0x529` both read 1 on the M1, which is why the int4 palette and the sparse form stream on the M1, while int8 and blockwise fold, a mechanism chapter 25 develops.
The compiler builds the table for a target by calling that target's constructor, so a single host recovers the table for every chip in the line whether or not it is the chip that is running.

## Per-family scalar matrix

The scalar parameters across the generation anchors show the same pattern: a value holds for a span of generations and then steps once, as [Table](#tbl:c24-scalar-matrix) gives across the generation anchors.

| Field (offset) | M1 (H13) | A14 | A15 | A16 (M4) | A17 (M5) |
| --- | --- | --- | --- | --- | --- |
| `num_nes` (`0x238`) | 4 | 4 | 4 | 4 | 16 |
| `max_operand_bytes` (`0x1b8`) | 2 MB | 2 MB | 2 MB | 2 MB | 2 MB |
| `max_tensor_width` (`0x138`) | 16384 | 16384 | 16384 | 65536 | 65536 |
| `max_tensor_depth` (`0x158`) | 16384 | 16384 | 16384 | 65536 | 65536 |
| `max_large_conv_kernel_dim_z` (`0x70`) | 16 | 16 | 16 | 16 | 16 |
| L2-resident threshold (`0x1f0`) | 0 | 0 | 32768 | 262144 | 262144 |
| instruction alignment (`0x218`) | 256 | 16 | 16 | 16 | 16 |
| reduction-transpose extent (`0x3f0`) | 192 | 192 | 384 | 384 | 384 |
| interchange-format count (`0x668`) | 3 | 13 | 16 | 14 | 14 |

Table: The scalar parameters at the generation anchors. {#tbl:c24-scalar-matrix}

The base-name M5 reads `num_nes` of 16 because the column is the 16-core Pro-class profile, while the base A17 profile has 4.
The per-die sequence runs 4 for the base name, 8 for the `g` suffix, 16 for `s` and the legacy 16-core profile, 32 for `c`, and 64 for the `d` Ultra-class die.

## Kernel-memory split

The streaming master byte does more than gate the compressed-weight stream: it selects which of two kernel-memory caps a layer's weights are sized against.
The legalization check is two lines of logic, reading a streamable flag and the master byte to pick the offset of the cap, then comparing the demand against it, as [listing](#lst:c24-kmem-split) gives.

```python
# ExceedKmemSizeLimit: split-legalize a layer's weights when they exceed the cap
def exceeds_kmem(hal, demand, is_streamable):
    cap = hal[0x210] if (is_streamable and hal[0x48f]) else hal[0x200]
    return cap < demand   # 0x200 = 64 KB dense, 0x210 = 16 MB streamed
```

Listing: The kernel-memory split, where a streamable weight under the streaming master is sized against the 16 MB cap and a dense weight against the 64 KB cap. {#lst:c24-kmem-split}

An ordinary non-streamed weight over 64 KB, or any weight over 16 MB, is thus split into multiple sub-layers on the M1, which raises the dispatch count and the compile time.
A streamed compressed weight is sized against the 16 MB cap and has far more weight per layer.
This is the weight path; it does not bound the activations, which stay within the maximum-tensor-dimension caps, so a layer with a tiny weight and a large activation is bounded by tiling cost in the partition passes rather than by this discrete limit.

## Dead and family-gated fields

Not every per-target value in the table is a live gate.
A byte-granular re-diff of all 28 target blobs leaves zero undecoded scalar fields, but several offsets that vary by family are populated by the per-chip builder and never read back through the table pointer, so their value is a write-only mirror.
Five scalar offsets are dead as table fields in this fashion: the global element cap at `0x18`, kernel-depth constant at `0x80`, legacy tiling granule at `0x260`, offset at `0x320`, and die-class flag at `0x29c`.
Each varies meaningfully by family, but the value a reader consumes is read off a different object that shares the byte displacement, a tensor-dimensions, compiler-parameters, or memory-pools structure, not the table.
The distinguishing test is whether the base register at the access holds the table pointer, since the same displacement aliases dozens of other by-reference structures, so a raw displacement match inside a table-typed function is not proof of a table read.

The one offset that looks dead on the M1 but is not is the FIFO-mode byte at `0x563`: it reads 0 on the M1 and is read through the table pointer under a branch that is taken only when the byte is set, which happens on the A18 generation.
A per-family value pattern alone does not establish a live gate; only a traced reader off the table base does.

## Naming the remaining capability flags

The capability bytes and the scalar limits are both fields of one struct, `ZinIrHalParameters`, the per-family blob the compiler builds for its target.
The struct has no per-field getter method, so the compiler reads a field through an inlined `ldrb` or `ldr` off the table pointer at a fixed offset, which is why a first pass recovers offsets and values but not names.
The names survive in one place only.
The compiler retains full mangled C++ symbols, and a reader function whose signature has `ZinIrHalParameters const&` reads each field, so the reader's name labels the field it reads.
A read is attributed to the table only when the load's base register is the function's `ZinIrHalParameters const&` argument, since the same byte displacement aliases dozens of other by-reference structures.

Cross-referencing the unnamed offsets against these reader functions names 95 more of them, of which roughly 30 resolve to a precise individual meaning with the base register verified against the table argument, the recovered names [Table](#tbl:c24-named-flags) attributes each to its reader function.

| Offset | Field | Reader function |
| --- | --- | --- |
| `0x4a8` | PE work-unit-shape supported | `PERasterization::ComputeWUShape` |
| `0x4ac` | small-source-mode compression supported | `ZinANELayer::AllowCompressionBasedOnSmallSourceMode` |
| `0x4b0` | non-power-of-2 work-unit width supported | `NERasterization::CanUseNonPowerOf2WUs` |
| `0x4f0` | preferred kernel layout format | `ZinIrKernel::GetPreferredKernelLayoutFormat` |
| `0x500` | transpose and multicast configuration | `ZinNELayer::FindValidMirInfoForTransposeCore` |
| `0x520` | secure-mode cache-hint DSID gate | `GetDSIDFromPriorityHalAndSecureMode` |
| `0x52c` | tensor-format support flag, pairs with the named `0x52d` fp8 byte | `ZinLayerValidationUtils::ValidateFormat` |
| `0x54c` | cache-prefetch kernel-task-interval limit | `ZinValidateTd<17>::ValidateCachePrefetchKernelTaskInterval` |
| `0x5a8` | cache-hint DSID value | `GetDSIDFromPriorityHalAndSecureMode` |
| `0x708` | reflective-padding maximum extent | `ZinValidateTd<20>::ValidateReflectivePaddingMode` |
| `0x748` | gather and texture-engine descriptor pointer | `ZinGatherLayer::CreateTELayer` |
| `0x8b4` | tile-height-errata threshold | `ZinTileHeightErrata::Workaround` |
| `0x8bc` | chaining enabled | `ZinIrRegAllocUtil::IsChainable` |
| `0x8e0` | kernel-caching enabled | `ZinIrTdValidationUtil::ValidateKernelCaching<N>` |

Table: Precise capability-flag names recovered this round, each attributed to its reader function. {#tbl:c24-named-flags}

The remaining 95-minus-30 additions are class-named: the reader identifies the subsystem the field gates without the exact semantics.
Examples are the per-axis DMA range bounds read by `ZinValidateTd<N>::CheckInRangeDmaAccess` and the texture-engine plane-equation coefficients in the `0x820` to `0x8f8` block gated by the named `0x81d` texture-engine byte on A14 and later.
Two candidates were rejected as table fields despite matching a displacement inside a table-typed function: `0xcf8` loads off an `adrp`-formed read-only constant rather than the table, and `0x678` loads off a nested object two pointers deep.
The same base-register test that found the five dead fields above also rules these out.

With this round the silicon-capability subset of the packed bitfield, the part the compiler reads to gate a feature per family, is fully named.
The struct is `0x938` bytes, and the few entries inside it that are not capability flags are the cost-model coefficient block at offsets `0x580` through `0x7f0`.
This block holds the frequency-to-efficiency curve, rate indices, performance multiplier that the roofline of chapter 18 reads, and about two soft fp64 coefficients.
These are all performance coefficients rather than legality caps.
A small number of capability fields are also true holdouts, read only off aliased bases.
The offsets past `0x938` hold no table: an earlier reading that located an A12 operation-emulation catalog at `0xa30` through `0xe84` was a read into the adjacent zeroed memory beyond the struct, not a real field.

## Attested is not reachable

A capability recorded in the HAL table attests support at the layer that reads the table.
It does not by itself prove that the operation lowers to a task descriptor and runs on the silicon.
These are distinct layers, and a capability present at the first can fail at the second.

The case that fixes the rule is three-dimensional convolution.
The HAL scalar at offset `0x70` records a 3D-conv kernel depth of 16 on the M1, attesting that the kernel geometry is permitted, and the compiler frontend recognizes the operation.
It still fails backend lowering on every device mask, returning the message that it is not supported on any backend.
The capability is in the table and the operation does not run.

The gap appears in the other direction as well, where a checker accepts an operation the code generator rejects.
On the M1 the top-k, sort, and dynamic-slice validators are all callable and all three are refused at code generation.
A bit in the table, a frontend that recognizes an operation, or a validator that passes are each a claim about one layer; only a compile-and-run on the target confirms the operation at the layer that executes it.
This is why the reachable surface of chapter 4 is smaller than the surface the table advertises, and why each native entry there was compiled and run on the M1 rather than inferred from a capability byte.
