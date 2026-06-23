# 25. Compression internals

> Every weight blob has a format enum, an integer one through thirty-one, that keys five parallel helper tables controlling all packing; the palette range is 7 through 27 and the unity range is 28 through 30.
> The engine reconstructs four compression forms: per-tensor and per-channel int8 affine, int4 lookup-table, structured sparsity, and blockwise affine, each a distinct dequantization chain the converter decodes into a kernel format.
> Two gates decide whether a compressed weight streams in its compressed bytes or folds to a dense half-precision constant: the kernel-streaming master at offset `0x48f` and the per-format palette and affine cluster at `0x520` through `0x539`.
> On the M1 only the int4 lookup-table and structured-sparsity forms stream; int8 and blockwise fold to dense through the dequantize-to-dense path.

Chapter 7 stated which compressed weight forms reach the engine and which save bandwidth.
The mechanism beneath that account follows: how a weight tensor becomes the compiled weight blob, what codec each form holds, and how the compiler decides whether the compressed bytes stream to the engine or fold to a dense half-precision constant first.
The path runs from a high-level dequantization operation chain down to the per-output-channel-group records the firmware re-bases into on every dispatch.
Bit-layout and address detail are M1/H13, table-descriptor codegen version five (family two, A13), unless another version or family is named.

## Kernel-format enum and its tables

Every weight blob has a format enum that names its storage encoding, an integer in the range one through thirty-one.
Five parallel tables keyed on that enum control all packing, and two structural ranges fall out of their guards and recur throughout the pipeline.
[Table](#tbl:c25-helpers) names the five kernel-format helper functions and what each one returns.

| helper | returns |
| --- | --- |
| `_ZinKernelFormatGetBitDepth(f)` | the index bit-width per element |
| `_ZinKernelFormatGetPaletteFormat(f)` | the palette sub-format, valid for `f` in 7 through 27 |
| `_ZinKernelFormatGetUnderlyingType(f)` | the underlying scalar type, int8, uint8, e4m3, or half |
| `_ZinKernelGetPaletteLUTSize(f, n)` | the per-codebook byte stride times the codebook count |
| `ZinKernelFormatGetTypeno(f)` | the type number written into the descriptor |

Table: The kernel-format helper functions keyed on the format enum and what each one returns. {#tbl:c25-helpers}

The palette formats are the enum range 7 through 27, selected by the guard `f - 7 < 0x15`.
The unity formats, the identity and scale-only weights that skip the multiply array, are the enum range 28 through 30, selected by `f - 0x1c < 3`.

A compiled weight unit is a fixed set of sections, and the descriptor that names them is the authoritative field list.
The descriptor holds an activation lookup table, palette lookup table, output-channel-group table, per-output-channel scale vector, per-output-channel bias vector, and the packed weight or index blob, the six-section field list [listing](#lst:c25-kernel-unit) reproduces.

```text
t%u = s%lu
  lut:     ; activation LUT     (typeno, bitdepth, lut_bytes)
  pal:     ; palette LUT        (palette_format, bitdepth, num_luts, lut_bytes)
  ocgs:    ; output-channel-group table  (the per-OCG records)
  scale:   ; per-output-channel scale    (half or single precision)
  bias:    ; per-output-channel bias
  weights: ; the packed weight or index blob
```

Listing: The per-output-channel-group kernel-unit descriptor, the authoritative six-section field list emitted into the compiled weight section. {#lst:c25-kernel-unit}

The palette table tuple holds the palette format, index bit-depth, codebook count, and codebook byte size; a codebook count above one marks the vector-palettized case.
The activation table and the palette table are peers in one descriptor and share a single on-chip-memory budget.

The element types the descriptor writes for a weight section are a subset of the 24-entry catalog.
They are int8 as type 2, uint8 as type 3, half-precision as type 5, the lookup-table tag as type 8, 4-bit unsigned as type 9, the 8-bit floating form as type 16, and 4-bit signed as type 23.
The half-precision type holds the codebook entries, scale, and bias, so a palette codebook supplies the multiply array with no further conversion.

## Four codecs

[Table](#tbl:c25-codecs) gives the four weight-compression codecs, each with its dequantization relation, on-device representation, the family it streams on natively, and the compiler gate that selects the stream.

| form | dequant relation | representation | streams natively on | gate |
| --- | --- | --- | --- | --- |
| int8 per-tensor / per-channel affine | $w = s(q - z)$, $z = 0$ on M1 | int8 byte stream plus per-channel half-precision scale | A14 and later | folds on M1 via the dequantize-to-dense path |
| int4 lookup-table | $w = \mathrm{LUT}[g/v][k][c \bmod v]$ | four-bit index stream plus sixteen-entry half-precision codebook | M1/H13 and later | streaming master plus the kernel-stride enable |
| structured sparsity | scatter of packed values into the mask | one-bit mask plus packed half-precision nonzeros | M1/H13 and later | streaming master, as a separate mask-and-values operand |
| blockwise affine | $w = s_b\,q$ per block | int8 byte stream plus per-block half-precision scale | A15 and later | folds on M1 and M2 via the dequantize-to-dense path |

Table: The four weight-compression codecs, each with its dequantization relation, native-stream family, and selecting compiler gate. {#tbl:c25-codecs}

Per-tensor and per-channel int8 use the affine form.
The stored byte is dequantized as

$$w = s\,(q - z)$$

with quantized byte $q$, half-precision scale $s$, and zero point $z$, the encode being $q = \mathrm{round}(x / s) + z$.
The scale is scalar or one value per output channel.
The M1 forces the zero point to zero, so the form is symmetric and reduces to $w = s\,q$: the asymmetric setter hard-asserts on the version-five table descriptor, and the firmware invariant requires the asymmetric-quantization configuration bit clear.

The per-output-channel scale and bias do not cost a runtime operation: the compiler folds them into the weight coefficients at compile time and holds them in the descriptor's scale and bias arrays as per-output-channel vectors.
The engine applies them at the gain-offset-control stage as $y = s_{\mathrm{out}}\,(\sum_k w'_k x_k) + b_{\mathrm{out}}$ where the coefficient already absorbs the dequantization scale.
This is why a symmetric quantized convolution on the M1 costs nothing extra for the affine: the compiler folds the scale and the zero point into the stored weights and the per-output-channel arrays before the dispatch.

The int4 lookup-table form stores a four-bit index per element into a sixteen-entry half-precision codebook.
Reconstruction is a table lookup with no arithmetic, since the codebook entries are already half-precision and supply the multiply array directly. The reconstructed weight is

$$w = \mathrm{LUT}\big[g / v\big]\big[\,k\,\big]\big[\,c \bmod v\,\big]$$

with output-channel-group index $g$, palette vector size $v$, the four-bit index $k$ for the element, and channel position $c$.
For the common per-tensor case the vector size is one and a single codebook holds all sixteen entries, so the relation flattens to one table lookup.
Four-bit weights have no affine path at all: there is no four-bit underlying scalar type in the type table, so a four-bit weight can only be a palette index.
That is why the lookup-table form is structurally a palette and is the one form that streams on the M1.

The palette index width is 2, 4, 6, or 8 bits, but the M1-practical widths are 4-bit, a 16-entry codebook, and 8-bit, a 256-entry codebook.
The 1-bit and 2-bit widths are rejected on the general path, and the 3-bit and 6-bit widths are version-gated to a later table descriptor, while the quantized-palette combination is capped below 256 entries.
A legality mask constrains the vector size, the count of consecutive output channels that share one codebook, to the set $\{1, 2, 4, 5\}$.
The format selector tests $2^{v}$ against the value `0x36`, which is `0b110110`, so vector sizes 0, 3, and any value of 6 or more are illegal.
A 4-bit codebook is 32 bytes and an 8-bit codebook is 512 bytes per lookup table, and the palette lookup table and the activation lookup table share one on-chip-memory budget bounded by the per-target palette-lookup-table size field.

Structured sparsity stores a one-bit nonzero mask plus the packed half-precision values of the surviving nonzeros.
The mask costs one bit per element and the values cost two bytes per survivor, so a weight that is half zeros or more stores well below its dense size.
Reconstruction walks the mask, consuming one packed value for each set bit and emitting a zero for each clear bit, exact apart from the half-precision rounding of the kept values.
Blockwise affine assigns a separate scale to each contiguous block of elements, finer than a per-channel scale and so lower in quantization error.

[Listing](#lst:c25-recon) gives the three reconstruction codecs in pseudocode, each the exact relation the engine applies for the affine, lookup-table, and structured-sparsity forms.

```python
# int8 / uint8 affine dequant (constexpr_affine_dequantize)
def dequant_affine(q, scale, zero_point):     # zero_point forced to 0 on the M1
    return scale * (q - zero_point)           # symmetric M1 form: scale * q

# int4 lookup-table reconstruction (constexpr_lut_to_dense)
def dequant_lut(index, codebook, g, vector_size):
    return codebook[g // vector_size][index][g % vector_size]   # already fp16, no arithmetic
    # per-tensor case: vector_size == 1, one 16-entry codebook -> codebook[0][index][0]

# structured-sparsity scatter (constexpr_sparse_to_dense): 1-bit mask + packed fp16 nonzeros
def densify_sparse(mask_bits, nonzeros, n):
    out = [0.0] * n                           # mask bit 1 = keep (nonzero)
    j = 0
    for i in range(n):
        if mask_bits[i]:                      # LSB-first packed, one bit per element
            out[i] = nonzeros[j]              # consume the next packed fp16 value
            j += 1
    return out
```

Listing: The three reconstruction codecs in pseudocode, the affine dequantization, lookup-table reconstruction, and structured-sparsity scatter. {#lst:c25-recon}

The sparse layout on the wire is the bitmask blob followed by the packed values blob, the operand layout [listing](#lst:c25-sparse-layout) gives.

```python
constexpr_sparse_to_dense operand layout
  mask     : ceil(n / 8) bytes   # 1 bit per element, LSB-first, dtype code UINT1 (9)
  nonzeros : 2 * popcount(mask) bytes   # fp16 survivors in scan order, dtype code FP16 (1)
  # streamed size ~ (1/16 + density) x the dense fp16 size
```

Listing: The on-the-wire operand layout of the structured-sparsity form, the bitmask blob followed by the packed half-precision nonzeros. {#lst:c25-sparse-layout}

## Stream-versus-fold decision

Two cooperating gates decide whether a compressed weight reaches the engine in its compressed bytes or is materialized to a dense half-precision constant before the dispatch.
The master gate is the kernel-streaming check.
It returns false unless the hardware-abstraction-layer streaming master bit at offset `0x48f` is set, which holds from the A13 generation onward.
It then requires the format be primary or convertible to half-precision by the direct-memory-access engine, plus unit stride on all axes, no dilation, no tile overlap, and an immutable weight: a mutable weight, a trained-in-place parameter, cannot stream.

A second gate guards the palette path.
The native lookup-table form streams only when the weight is vector-palettized and the per-format kernel-stride-enable bit at offset `0x529` is set.
When both hold, the compiler sets the palette-enable and stream flags and zeroes the fold sub-fields.
On the M1 the offset `0x529` bit is set, so the palette stream is live while the other formats fold.

Structured sparsity reaches the engine through the master gate as a separate operand rather than through the palette path.
The sparse weight lowers to a mask producer and a nonzero-values producer held as their own blobs.
Those stream under the master at offset `0x48f`, which is set on the M1, not under the per-format palette and affine cluster at offsets `0x520` through `0x539`, which is clear on the M1.
A live byte comparison confirms the form: the sparse weight blob stores the one-bit mask and the half-precision nonzeros at about 0.43 times the dense size, with no dense half-precision constant present, whereas a fold would show a single full-width half-precision blob.

The int8 and blockwise forms have no streamed encoding on the M1.
The converter builds the native quantized kernel, but the lowering routes it through the dequantize-to-dense path, which reconstructs the weight to a dense half-precision constant before the direct-memory-access transfer.
The stored bytes are then plain half-precision and move at full width, so a weight-streaming-bound layer gets no bandwidth gain from these forms on the M1.

A budget relaxation accompanies the stream decision.
A streamed weight is sized against the relaxed on-chip-memory cap at offset `0x210`, while a folded or dense weight is sized against the dense cap at offset `0x200`, which is sixty-four kilobytes on the M1.
The compressed path thus has far more weight per layer.

## Per-output-channel-group packing

The weight section is a sequence of per-output-channel-group records, and each record is the fixed 14-word block [Table](#tbl:c25-ocg-record) details word by word with its computed offset and size and its ten unrelocated address slots.

| Word | Field | Initial value |
| --- | --- | --- |
| `[0]` | output-channel-group byte offset into the section | computed |
| `[1]` | output-channel-group size, the channel count for this engine | computed |
| `[2]` through `[0xb]` | ten address relocation slots | `-1` |
| `[0xc]`, `[0xd]` | reserved flags | 0 |

Table: The per-output-channel-group record, the 14-word block with its computed offset and size and its ten unrelocated address slots. {#tbl:c25-ocg-record}

The ten slots left at `-1` are the per-record device addresses the loader patches in: the input and output tile bases, per-buffer coefficient sub-buffer bases, and four coefficient-stream bases for bias, post-scale, palette lookup, and activation lookup.
Those four streams map one-to-one to four kernel direct-memory-access sub-channels in the kernel-and-common register group, each with the enable, base-offset, and relocation register [Table](#tbl:c25-dma-streams) gives.

| Descriptor slot | Enable register | Base-offset register | Relocation register |
| --- | --- | --- | --- |
| bias | `0x5548` | `0x554c` | `0x1554` |
| post-scale | `0x5558` | `0x555c` | `0x1558` |
| palette lookup | `0x5568` | `0x556c` | `0x155c` |
| activation lookup | `0x5578` | `0x557c` | `0x1560` |

Table: The four kernel direct-memory-access coefficient streams, each with its enable, base-offset, and relocation register. {#tbl:c25-dma-streams}

There is no zero-point stream, since the M1 symmetric form folds the zero point to zero; the per-output-channel scale and bias streams supply the dequantization scale instead.
The M1 replicates the kernel coefficients per engine core, one copy per core, because it lowers to the per-core layout rather than the shared kernel-memory layout that the A14 generation and later use.
A four-core M1 thus has four per-core records each with its own output-channel-group vector.

The activation lookup table shares the descriptor and the budget with the palette lookup table.
It is a 43-entry half-precision record: two input-clamp values, a mode flag, a scale, the 33 knot values uniform in the input domain, four tail extrapolation coefficients, and a packed mode word.
At runtime the input maps affinely onto the 32 segments between the 33 knots, the integer part selects a segment, and the fraction controls a half-precision linear interpolation, with the output clamping to the end-knot beyond the input-clamp domain.
The activation lookup table and the palette lookup table are co-located in the weight section and located by one offset cursor, which is why they draw on one shared budget.

## Sparsity datapath

The engine has two independent sparsity mechanisms.
The first is compute-time zero-skip.
The table descriptor has a detect-zeros bit that is implemented on the M1, and when it is set the multiply array skips a multiply-accumulate whose weight is zero.
The cost model scans the weight values for their zero density once, caches the ratio, and reduces the convolution cycle estimate by it, adding the implicit zeros that strided and padded convolutions contribute.
This mechanism is format-independent: it fires on any kernel with zeros regardless of storage encoding, including weights that fold to dense for storage.
On a bandwidth-bound stack the zero-skip alone moves the limit by about one percent, because skipping multiply-accumulates does not help a layer limited by the weight stream rather than by the array.

The second is the sparse-binary store with on-chip decompress, which is the source of the bandwidth gain.
The mask and the packed nonzeros cross the direct-memory-access stream, and the engine decompresses them on chip into the dense tile.
The on-device kernel-configuration register holds a single packed sparse-format field beside the palette-enable and palette-bits fields in the same word.
A six-by convolution stack at sixty-three percent zeros streams the sparse form at about 0.43 times the dense weight bytes and runs 1.55 to 1.64 times faster than the same weights stored dense.
Effective bandwidth rises from about 29 to about 48 gigabytes per second, which is the M1 weight-stream ceiling.
The compiled program is byte-identical between the dense and sparse bundles at 2416 bytes, so the difference is entirely in the streamed weight payload and its descriptor, which is the signature of a native stream rather than a fold or a changed program.

The version-five table descriptor asserts that sparse-binary mode is not supported, and this assert reconciles with the live stream because it governs a different path.
That assert bounds the sparse packing of the palette-index plane inside the compiled weight blob, reachable only through the vector-palette flag, so a non-palettized weight never reaches it.
The streamed sparse weight is the separate mask-and-values operand described above, which does not pass through that table-descriptor bit.

## fp8 datapath

The compiler has a complete fp8 datapath that no M1 generation can reach.
The 8-bit floating form is two distinct things in the binary: the E4M3 form, a gated hardware weight and activation format with element code `0xc`, and the E5M2 form, a conversion format with element code `0xd`.
E4M3 is the gated one: its kernel-format validity admits it as a native weight element type, but the encoder that writes its element code exists only from the A17 generation.
The runtime capability byte at `0x52d` that enables the format is set on the A18 generation alone.
On the M1 the format register is two bits wide with no E4M3 codepoint at all, so passing the format triggers a compile-time assert rather than a runtime refusal.
The accumulator stays at the wide half-precision-class width on every family, so the 8-bit form is an input and storage width supplying the same multiply array, not an accumulate width.
Its throughput runs on the same double-multiply path the int8 form uses.
The full fp8 datapath, format-register delta, E5M2-to-half fold asymmetry, and conversion functions are decoded in chapter 36.

## Codegen version matrix

The same kernel format produces different weight bytes per table-descriptor version, which is why the M1 and the M5 differ at the bit level rather than only in scalar limits, as [Table](#tbl:c25-version-matrix) records capability by capability.

| Capability | M1 version-five descriptor | Later descriptor |
| --- | --- | --- |
| sparse-binary index packing | asserts unsupported | a packed flag at descriptor offset `0x424` |
| multi-codebook palette | asserts unsupported | implemented |
| asymmetric quantization | asserts unsupported | implemented |
| zero-skip detect | implemented | implemented |
| palette enable and bit width | implemented, 4 and 8 bit | implemented, plus 3 and 6 bit |

Table: Codegen capability per table-descriptor version: the M1 version-five descriptor lacks the sparse-binary, multi-codebook, and asymmetric encoders. {#tbl:c25-version-matrix}

The M1 legal kernel-format space is thus the intersection of the format enumeration and the version-five descriptor: dense half-precision, palettized 4-bit and 8-bit including the single-codebook vector form, symmetric int8 that folds to half-precision, and the unity forms.
The sparse-binary index packing, multi-codebook palette, asymmetric quantization, fp8 form, and 3-bit and 6-bit palette widths are absent from the version-five codegen entirely, a missing encoder rather than a runtime refusal.

## Winograd and the compressed-weight interaction

The full Winograd eligibility gate is derived in chapter 20: the enable bit, kernel-axis and tile-size conditions, and the $\mathrm{OCG} \times K_y \times K_x \times K_d \times 2$ work threshold against its non-float, float, and packed floors.
The compression-relevant point is the weight-format interaction: eligibility requires a non-unity, non-sparse weight, so a sparse or unity-format weight is excluded from the Winograd path and keeps its own datapath.
The transform matrices $G$, $B^\top$, and $A^\top$ are resident in the array rather than stored in any shippable weight blob, so no codec holds them.
