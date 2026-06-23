# 7. Weights and compression

> The engine reconstructs four compressed weight forms at the multiplier input, and on the unentitled direct route a form either streams its compressed bytes for a bandwidth gain or folds to dense fp16 for none.
> Which outcome applies to a form is set by the target chip: the M1 streams only int4 lookup-table at 2.37 times fp16 and structured sparsity at 1.55 to 1.64 times, and folds int8 and blockwise affine.
> The M5 streams all four at 1.6 to 1.8 times.
> Choose the form by what streams on the target, and on the M1 fall back to int8 only where its halved stored size pays, since the int8 fold expands to fp16 in DRAM and yields no bandwidth there.

A compiled network holds its weights as a separate stream that the engine reads from DRAM on every dispatch.
The weight stream is the primary cost of any layer whose arithmetic intensity is low, which covers most decode-shaped and projection-heavy work.
Compressing the weights moves fewer bytes across that stream; it does more than shrink a stored file.

## Compression is a bandwidth feature on the direct route

Compressed weights reach the engine through the direct runtime described in chapter 5.
They are not gated behind an entitlement, and the operations that reconstruct them are accepted by the compiler without special privilege.
The reconstruction happens at the multiplier input, consistent with the fp16 datapath of chapter 3: a compressed weight is turned back into fp16 before it reaches the multiply array, and the multiply that follows is the same fp16 multiply as for an uncompressed weight.

The word compression covers two distinct outcomes.
A form that streams reaches the engine in its compressed bytes and is decompressed on chip, so fewer bytes cross DRAM and the layer runs faster when it was bandwidth bound.
A form that folds is reconstructed to a dense fp16 constant before the dispatch, so the bytes that cross DRAM are full-width fp16 and the layer gets no bandwidth gain.
The target chip sets which outcome applies to a form, not the frontend.

## Compression forms

The engine reconstructs four weight-compression forms, the same linear quantization, palettization, and pruning the conversion tools document as a model-size feature [AppleCoreMLTools].
Per-tensor and per-channel int8 hold the weight as a single byte per element with an fp16 scale, one scale for the whole tensor or one per output channel, reconstructed as the scale times the byte.
The reconstruction is the affine dequantization

$$w = s\,(q - z)$$

with stored quantized byte $q$, fp16 scale $s$, and zero point $z$, the encode being $q = \mathrm{round}(x / s) + z$.
The zero point folds to zero on the M1 generation, so the int8 form there is symmetric and the relation reduces to $w = s\,q$ with $z = 0$, as [listing](#lst:c7-int8-dequant) shows at the multiplier input alongside the symmetric form.

```python
# affine int8 dequant, evaluated at the multiplier input
w_fp16 = scale * (q - zero_point)        # general affine form
w_fp16 = scale * q                       # M1: zero_point folds to 0, symmetric only
# q is int8 in [-127, 127]; scale is fp16, scalar or one per output channel.
```

Listing: The affine int8 dequantization evaluated at the multiplier input, with the symmetric form the M1 reduces to. {#lst:c7-int8-dequant}

Each compressed form is declared to the compiler as a single `constexpr_*` reconstruction op that folds into the weight descriptor rather than becoming a standalone backend operation, one per form as [listing](#lst:c7-constexpr-ops) names them.
The op count per form is fixed, and a violation is a hard validation error.

```python
# the constexpr reconstruction op per form (one MIL op each, folded into the conv/linear weight)
constexpr_affine_dequantize(q, scale, zero_point)   # int8 affine:  w = scale * (q - zero_point)
constexpr_lut_to_dense(indices, lut)                # int4 palette: w = lut[indices]
constexpr_sparse_to_dense(mask, nonzeros)           # sparsity:     scatter into mask positions
constexpr_blockwise_shift_scale(q, scale)           # blockwise:    one scale per contiguous block
```

Listing: The single reconstruction op that declares each compressed weight form to the compiler. {#lst:c7-constexpr-ops}

The int4 lookup-table form holds a four-bit index per element into a sixteen-entry fp16 codebook, and reconstruction is a table lookup with no arithmetic since the codebook is already fp16.
Structurally it is a palette rather than an arithmetic dequantization, which is why it behaves differently from int8 on the streaming question.

A worked example shows the packing and the decode.
The four weights `[1.0, 0.0, 0.0, 1.0]` index a sixteen-entry fp16 codebook whose first two slots hold `0.0` (entry 0, `0x0000`) and `1.0` (entry 1, `0x3c00`).
The index stream `[1, 0, 0, 1]` packs two four-bit indices per byte, low nibble first, so the four weights occupy the two bytes `0x01` and `0x10`.
Decoding reads each nibble as a table index, recovering `[1.0, 0.0, 0.0, 1.0]`.

Structured sparsity holds a one-bit mask marking the nonzero positions plus the packed fp16 values of those nonzeros.
The mask costs one bit per element and the values cost two bytes per surviving element, so a weight that is half zeros or more stores well below its dense size.
Reconstruction scatters the values back into the masked positions exactly apart from the fp16 rounding of the kept values.
Blockwise affine holds a separate scale for each contiguous block of elements, finer than a per-channel scale and so lower in quantization error.

## What streams on each family

[Table](#tbl:c7-streams) gives, for each compressed weight form, whether it streams or folds to dense on each chip generation, with the measured speedup.

| Form | M1/H13 | A14/M2 | A15/M3 and later | M5/H17s | Speedup |
| --- | --- | --- | --- | --- | --- |
| int8 (per-tensor / per-channel) | fold | stream | stream | stream | M5 1.6-1.8x; M1 folds, no stream gain |
| int4 lookup-table | stream | stream | stream | stream | M1 2.37x; M5 1.6-1.8x |
| structured sparsity | stream | stream | stream | stream | M1 1.55-1.64x at 0.43x dense bytes; M5 1.6-1.8x |
| blockwise affine | fold | fold | stream | stream | M5 1.6-1.8x; M1 and M2 fold, no stream gain |

Table: Whether each compressed weight form streams or folds to dense per chip generation, with the measured speedup. {#tbl:c7-streams}

On the M1, of the H13 generation, only the int4 lookup-table form streams natively.
A bandwidth-bound stack of one-by-one convolutions runs 2.37 times faster with int4 weights than with fp16, measured on the M1, because the four-bit indices move at a quarter of the fp16 byte count.
Structured sparsity also streams natively on the M1: a convolution stack that is about sixty-three percent zeros runs 1.55 to 1.64 times faster than the same weights stored dense, at 0.43 times the dense weight bytes.
The output is bit-faithful to the dense reference apart from fp16 rounding.

The int8 and blockwise forms fold on the M1.
The accuracy cost is the per-output int8 quantization: the int8 weight tracks the fp16 result at a cosine near 1.0, with a relative error near one percent against an fp32 reference where fp16 is near two parts in ten thousand.

From the A14 generation the int8 and sparse forms also stream, alongside int4, measured on the M2 of the H14 generation at 0.64 and 0.54 times fp16 on a bandwidth-bound matmul.
On the A14 and M2 the int8 weight is dispatched as int8 and reconstructed from half the bytes.
The measured latency runs from 0.85 times fp16 at a two-thousand-wide weight down to 0.52 times fp16 at an eight-thousand-wide weight, deeper as the weight grows and the stream dominates.
Blockwise affine does not stream on the A14: it still folds there, measured at 0.985 times fp16, a near-zero bandwidth gain, and it first streams from the A15 generation.
On the M5, of the H17s generation, all four forms stream, at a measured 1.6 to 1.8 times fp16 on bandwidth-bound layers.

The streaming-versus-folding split is a hardware-abstraction-layer decision, not a property of any single reconstruction operation.
Every weight-bearing operation is legal on every chip, so the boundary is in a set of feature bytes the compiler reads from the per-chip table.
In that table one master byte enables weight streaming at all and a cluster of per-format bytes admits each compressed form by generation.
[Table](#tbl:c7-hal-bytes) gives those feature bytes and the generation each switches on.

| Hardware-abstraction-layer byte | Role | M1/H13 | A14/M2 | A15/M3 | A16, A17, M5 |
| --- | --- | ---: | ---: | ---: | ---: |
| `+0x48f` | kernel-streaming master | 1 | 1 | 1 | 1 |
| `+0x528`, `+0x532`, `+0x537` | per-format gates that switch on at A14 | 0 | 1 | 1 | 1 |
| `+0x520`, `+0x523`, `+0x533`, `+0x539` | per-format gates that switch on at A15 | 0 | 0 | 1 | 1 |
| `+0x529` | palette and stride gate | 1 | 1 | 1 | 1 |

Table: The hardware-abstraction-layer feature bytes that gate compressed-weight streaming, by chip generation. {#tbl:c7-hal-bytes}

The master byte is set from the A13 generation, which is why the M1 streams anything at all, and the palette gate is set on the M1, which is why the int4 lookup-table form streams there.
The per-format gates for the affine int8 and blockwise forms are clear on the M1 and switch on at the A14 and A15 generations, which is why those two forms fold on the M1 and stream on the newer parts.
Structured sparsity streams on the M1 by a separate route: it is held as a mask-and-values weight operand under the master byte rather than as a palettized kernel coefficient, so it streams under the master gate independent of the per-format cluster.
The int8 floor at A14 is confirmed on the M2 silicon, and the blockwise floor at A15 is read from that gate pattern and not yet confirmed on the intermediate silicon.

On the M1 the int8 fold is a stored-size saving only.
The weight is half the size on disk, but it is expanded to a dense fp16 constant in DRAM before the data-movement step, so a weight-streaming-bound matmul moves full-width fp16 bytes and runs at the fp16 latency, with no bandwidth gain.
The int8 weight first streams as int8 on the A14 and M2 generation, where it is dispatched as int8 and dequantized at the multiplier input rather than materialized to a dense fp16 constant in DRAM.
A matmul-path measurement on the A14 and M2 makes this concrete, and [table](#tbl:c7-int8-matmul) gives it: at a weight matrix wide enough to leave the dispatch floor, the int8 and fp16 weight streams reach the same effective bandwidth against their stored bytes, while the int8 form moves half the bytes.

| Weight width $K=N$ | fp16 latency | int8 latency | int8 / fp16 |
| --- | ---: | ---: | ---: |
| 2048 | 0.290 ms | 0.253 ms | 0.87 |
| 4096 | 0.740 ms | 0.447 ms | 0.60 |
| 8192 | 2.610 ms | 1.351 ms | 0.52 |

Table: The int8 matmul-path weight stream against fp16 on the A14 and M2, the latency ratio approaching one half as the weight grows. {#tbl:c7-int8-matmul}

## Choosing a form

What streams on the target drives the choice of compression form, chip by chip.
On the M1, prefer structured sparsity for a weight that is half zeros or more, since it streams and is lossless apart from fp16 rounding, and prefer the int4 lookup table otherwise, since it is the densest form that streams there.
Reserve int8 on the M1 for the case where sixteen levels are too coarse, taking it for its halved stored size, since the form folds to fp16 in DRAM and yields no bandwidth there.
On the M5 and the newer generations, where every form streams, choose by accuracy per byte rather than by what streams.

The wide accumulator of chapter 3 is what makes streamed low-precision weights safe on the layers that tolerate them.
A streamed weight is reconstructed to fp16 and then enters the same fp16 multiply and wide reduction as a dense weight, so the only precision lost is the quantization of the weight itself, not the accumulation.
On convolution, matrix multiply, and normalization, whose partial sums stay in range, the reduction holds and the compressed weight keeps the layer's accuracy.
A cancellation-heavy step has no fp16-safe form, compressed or not, for the reason given in chapter 3.

## Patching weights in a compiled program

A host can patch a compiled program's weights in place without recompiling.
Each weight tensor occupies a decoded region of the program image with a known tiling.
A convolution weight is in a `0xC0`-stride layout and a matrix-multiply weight in a `0x40`-stride layout, and editing the weight values leaves the program descriptor unchanged.
A host can thus swap new weights into an already-compiled program rather than rebuilding it, which makes a weight-only update inexpensive.

The driver has a per-patch-mutable-buffer accounting path that confirms the route, `ANEScheduler::pendingRequestsPerPatchMutableBuffer`.
The descriptor names the operations and binds the buffers, and a weight-value edit touches neither, so the same compiled program runs with the new coefficients.

## Automating the choice

The estimate classifies a layer as compute bound or bandwidth bound so the procedure chooses a form only where the layer is bandwidth bound and a stream would help.
The procedure keeps the smallest form that streams natively on the target and clears an accuracy tolerance against an fp32 reference.
A form that folds to dense fp16 moves the same bytes as fp16, so it cannot help a bandwidth-bound layer; only a streaming form does.
Which forms stream depends on the target: the M1 streams int4 and sparse while int8 and blockwise fold, and the A14 and later stream int8 as well.
Sparsity applies only when at least half the weight is zero, and the candidates are tried smallest-bytes-first, int4 before sparse before int8.
Here `accuracy_error` round-trips the weight through the candidate form and compares the layer output against the fp32 reference.

```python
tolerance = 0.01    # max relative error of the layer vs an fp32 reference

def native_streams(chip):
    if chip == H13:   return [int4, sparse]       # M1: int8 and blockwise fold
    if chip >= H14:   return [int4, sparse, int8]

def choose_weight_form(layer, weights, chip):
    if not is_bandwidth_bound(layer, chip):
        return fp16
    candidates = native_streams(chip)
    if fraction_zero(weights) < 0.5:
        candidates.remove(sparse)
    for form in sorted(candidates, key=bytes_per_weight):
        if accuracy_error(form, weights, layer) <= tolerance:
            return form
    return fp16

form = choose_weight_form(layer, W, chip=H13)     # M1 -> int4 (streams, 2.37x)
program = compile(graph_with(W, form), chip=H13)
```

On the M1 the procedure chooses a folding form, int8 or blockwise, only where the halved stored size or a finer scale pays on the matmul path, not for a stream gain.

## Reference: per-form streaming and measured cost

[Table](#tbl:c7-reference) collects the per-form streaming behavior and measured cost by chip generation.

| Constant | Generation | Value |
| --- | --- | ---: |
| int4 lookup-table stream speedup | M1/H13 | 2.37x fp16 |
| Structured sparsity stream speedup | M1/H13 | 1.55-1.64x fp16 |
| Structured sparsity stored bytes | M1/H13 | 0.43x dense |
| int8 fold on M1 (stored-size saving, no stream gain) | M1/H13 | 1.0x fp16 latency |
| int8 stored size on disk | M1/H13 | 0.5x fp16 |
| int8 accuracy versus fp32 reference | M1/H13 | cosine near 1.0, relative error near 1% |
| int8 matmul latency, 2k-wide weight | A14/M2 | 0.85x fp16 |
| int8 matmul latency, 8k-wide weight | A14/M2 | 0.52x fp16 |
| int8 stream speedup | A14/M2 | 0.64x fp16 |
| Structured sparsity stream speedup | A14/M2 | 0.54x fp16 |
| Blockwise affine fold | A14/M2 | 0.985x fp16 (no stream gain) |
| All four forms stream speedup | M5/H17s | 1.6-1.8x fp16 |
| Streaming master gate | A13 | on |
| int8 per-format stream floor | A14 | confirmed on M2 silicon |
| Blockwise affine per-format stream floor | A15 | predicted, not silicon-confirmed |

Table: The per-form streaming behavior and measured cost, by chip generation. {#tbl:c7-reference}

The compressed weight element types are a subset of the engine element-type catalog the kernel descriptor records.
The forms that bear on weight encoding are the integer lanes, fp16 dense and codebook entries, and packed palette indices.
[Table](#tbl:c7-element-types) gives those weight-relevant entries of the catalog with the width each has.

| Element type | Width | Role in weight encoding |
| --- | ---: | --- |
| int8 | 1 byte | affine int8 lane |
| uint8 | 1 byte | affine uint8 lane |
| float16 | 2 bytes | dense weight, codebook entry, scale, and bias |
| uint4 | 4 bits | 4-bit palette index |
| int4 | 4 bits | 4-bit signed, palette-only on the M1 |
| e4m3 | 1 byte | fp8, gated off on the M1 |

Table: The weight-relevant entries of the engine element-type catalog, with the width each has. {#tbl:c7-element-types}

There is no int4 arithmetic lane in the datapath, so the four-bit value is always a palette index into the sixteen-entry fp16 codebook, which is why the element-type table marks int4 palette-only on the M1.
