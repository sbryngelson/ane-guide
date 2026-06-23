# 17. Model-design rules

> A network compiles only when every operation is inside the validator limits: fp16 activations, width and height at most 16384, channel at most 65536.
> Convolution kernel width is capped at 13 and arg-min and arg-max reduce an axis of at most 2048.
> Keep a single operation's largest live operand under 2 MB or it tiles and streams from DRAM.
> Make channel counts a multiple of the interleave factor and group counts divide the core count, or the engine pads lanes and runs slower.

The validator limits are design rules: the shape, rank, dtype, and mode constraints the per-operation validators enforce, the working-set threshold that decides whether a tensor stays on chip or tiles, and the channel and kernel bounds that distinguish a fast layer from a padded one.
Chapter 4 says which operations exist; this chapter says the shapes those operations accept.

## Dtype rule

The datapath is fp16, so every operand a model presents to the engine is fp16.
The backend accepts the fp32, int32, and bf16 type annotations, but does not implement them as wider arithmetic, and they do not reach the silicon.
The M1 rejects a cast to int32, and bf16 is not usable as a program input or output dtype.
The one widened quantity is the matrix-multiply accumulator, which is internal and not a tensor a model declares.

Weights are the exception that still resolves to fp16 at the compute step.
The engine accepts an int8 weight and runs it through the quantized-convolution path, and the compressed weight forms in chapter 7 stream in their stored width.
Activation tensors that flow between operations stay fp16 regardless.

## Shape and rank limits

Tensor extent is bounded per axis, and the bounds are exact.
A tensor may span up to 16384 along width, 16384 along height, and 65536 along channel; one element past any of these rejects at compile time.

$$W \le 16384, \quad H \le 16384, \quad C \le 65536$$

These were measured on the M1 by sweeping each axis until compile flipped from accept to reject, and the boundary matched the decompiled maximum-dimension field exactly.
Width 16384 compiles and 16385 rejects, and the same one-element step holds for height and for channel.

Convolution has its own kernel bounds, narrower than the tensor bounds and set by the kernel-format field.
On the M1 the fp16 convolution kernel width is bounded at 13: a kernel of width 14 rejects.
Kernel height reaches 29 when the input is tall enough to hold it, and a kernel taller than its input rejects as a too-large kernel rather than as a height cap.
Stride, dilation, and padding do not reject on the M1.
A stride outside the native set decomposes to a large-stride path and still compiles, dilation up to 8 compiles, and padding up to 16 compiles.
The pooling window is far looser than the convolution kernel and accepts sizes of 128 and beyond, with a tile-alignment quirk that rejects specific odd windows near a power-of-two boundary.

Several operations cap a reduction axis for fp16 accuracy rather than for memory.
The arg-min and arg-max operations reduce a channel of at most 2048, and the spatial form caps height or width at 2048 the same way; 2049 rejects.
This 2048 limit is independent of the 16384 and 65536 tensor bounds and applies to the reduction axis alone.

## Mode and divisibility rules

The matrix-multiply backend operation requires the contracted depth to be one on both operands, and a depth greater than one rejects.
The two operands broadcast on batch and on height when one side is one, and the inner dimensions must agree so that the left width plus padding equals the right channel count.

Grouped convolution divides both the input and the output channel count by the group count, and a count that does not divide rejects.
For throughput, the group count should also divide the engine core count, which is four on the M1.
A group count of one, two, or four maps one neural-engine lane to one lookup-table lane.
A count that does not divide the core count loses that one-to-one mapping and runs slower.

Channel counts should be a multiple of the interleave factor.
The engine stores tensors channel-interleaved, and a channel dimension that is not a multiple of the interleave factor is padded out to one, leaving lanes unused.
The width axis aligns to a 16-byte direct-memory-access granule, the same factor of 16 that governs the slice-tiling path.
Interleave-aligned, power-of-two channel counts avoid silent padding.
[Listing](#lst:c17-interleave-conv) contrasts an interleave-aligned tensor with a padded one and rewrites a fully-connected layer as the equivalent one-by-one convolution.

```python
# Tensor order is [N, D, C, H, W]; the engine stores it channel-interleaved
# and aligns the last axis to the 16-byte DMA granule.

aligned = (1, 1, 64, 56, 56)   # channels first, W = 56 fills 16-byte lanes
padded  = (1, 1, 64, 56,  1)   # singleton last axis pads out to the granule,
                            # wasting every lane but the first

# Replace a fully-connected layer with an equivalent 1x1 convolution so it
# runs on the convolution datapath. A linear y = x @ W.T over C_in -> C_out
# is the same arithmetic as a 1x1 conv over a [N, 1, C_in, 1, 1] feature map.
def linear_as_1x1_conv(x, W):
    # W: [C_out, C_in] reshaped to a 1x1 kernel [C_out, 1, C_in, 1, 1]
    kernel = W.reshape(W.shape[0], 1, W.shape[1], 1, 1)
    return conv(x, kernel, strides=[1, 1], groups=1, kernel_sizes=[1, 1])
```

Listing: Channel-interleaved tensor layout and a fully-connected layer rewritten as an equivalent one-by-one convolution. {#lst:c17-interleave-conv}

Several mode constraints are family-gated and reject on the M1 specifically.
The symmetric and reflect padding modes need the texture engine and are unavailable on the M1, where they decompose or are refused.
A square-after-reduction fused mode is absent on the M1 and arrives on the A14.
The texture-engine sampling operations, resize as a hardware sampler, crop-resize, resample, and affine, are absent on the M1 and arrive on the A14, and the trigonometric sine and cosine arrive on the A15.

## Working-set rule

The decisive size rule is the on-chip working set.
A tensor that fits the on-chip static memory stays resident across the operation; a tensor that exceeds it splits into tiles that the engine streams one at a time.
The working set is the 2 MB on-chip region, and the matrix-multiply path also bounds the output-channel footprint against a 64 KB kernel-memory budget, rejecting a matrix multiply whose output channels do not fit.

Tiling preserves the result.
A reduction or transpose over an axis larger than the on-chip threshold switches to a tiled route at no change to the output, so the threshold is a performance boundary, not a correctness one.
Where latency matters, keep a single operation's largest live operand under 2 MB.
At fp16, 2 MB holds about $2^{21}/2 \approx 1.05 \times 10^{6}$ elements, so a square activation of side near 1024 is at the edge of the resident regime, and a larger one streams.
This is the working-set input to the cost model in chapter 18, where the roofline relation turns the resident-versus-streaming split into a latency estimate.

## Validating shapes against the design rules

The cost estimate reports the binding limit for each operation, so a shape violation or a working set above 2 MB shows before the program reaches the compiler.
[Listing](#lst:c17-validate) walks every layer of a graph against the rank, extent, kernel, group, interleave, and working-set rules and flags each violation before the build.

```python
# Tensor order is [N, D, C, H, W]; the engine stores it channel-interleaved.
# Check every layer against the design rules before building, target = H13.

for each layer in graph G:
    # Rank rule: at most 5 axes
    if rank(layer.output) > 5:
        flag(layer, "rank above 5 rejects at compile")

    # Per-axis extent rules
    if width(layer)   > 16384:  flag(layer, "width above 16384 rejects")
    if height(layer)  > 16384:  flag(layer, "height above 16384 rejects")
    if channel(layer) > 65536:  flag(layer, "channel above 65536 rejects")

    # Convolution kernel and group rules
    if layer is conv:
        if kernel_width(layer) > 13: flag(layer, "kernel width above 13")
        if (in_channel(layer) mod groups(layer)) != 0: flag(layer, "in-channels % groups != 0")
        if (out_channel(layer) mod groups(layer)) != 0: flag(layer, "out-channels % groups != 0")
        if (core_count mod groups(layer)) != 0: warn(layer, "groups do not divide cores, slower")

    # Reduction-axis rule for arg-min / arg-max
    if layer is argmin or layer is argmax:
        if reduced_axis_extent(layer) > 2048:  flag(layer, "arg reduction axis above 2048 rejects")

    # Channel-interleave rule: pad warning, not a reject
    if (channel(layer) mod interleave_factor) != 0:
        warn(layer, "channel not interleave-aligned: pads out, wastes lanes")

    # Width DMA-granule rule: last axis aligned to 16 bytes
    if (width(layer) mod 16) != 0:
        warn(layer, "width not aligned to 16-byte DMA granule: pads to granule")

    # Working-set rule: largest live operand under 2 MB stays on-chip, else it tiles and streams
    bytes = element_count(largest_live_operand(layer)) * 2     # fp16 = 2 bytes per element
    if bytes > 2 * MB:
        warn(layer, "working set above 2 MB: tiles and streams from DRAM (slower, still correct)")

# Reshape any flagged layer before tuning; address pad/tile warnings where latency matters.
```

Listing: Walking every layer of a graph against the design rules, flagging each reject and pad warning before the build. {#lst:c17-validate}

A graph that prints `validates` false, a working set above 2 MB, or nonzero channel padding is reshaped before any other tuning, because a rejected shape never reaches the silicon and a padded channel leaves lanes unused on every dispatch.

## Reference: the per-operation design rules

[Table](#tbl:c17-design-rules) collects every validator limit a model must satisfy, with each limit and the result of exceeding it.

| Constraint | Limit | Consequence if exceeded |
| --- | --- | --- |
| Activation dtype | fp16 only | int32 cast and bf16 input or output rejected on M1 |
| Width axis | 16384 | 16385 rejects at compile |
| Height axis | 16384 | 16385 rejects at compile |
| Channel axis | 65536 | 65537 rejects at compile |
| Conv kernel width (fp16) | 13 | 14 rejects at compile |
| Conv kernel height | 29, input permitting | kernel taller than input rejects |
| Arg-min/arg-max reduction axis | 2048 | 2049 rejects for fp16 accuracy |
| Matrix-multiply operand depth | 1 | depth greater than 1 rejects |
| Matrix-multiply output channels | 64 KB kernel-memory budget | rejects when output channels do not fit |
| Conv group count | divides input and output channels | non-dividing count rejects |
| Conv group count, for speed | divides core count (4 on M1) | one-to-one lane mapping lost, slower |
| Channel multiple of interleave | interleave-aligned | padded out, lanes unused |
| On-chip working set | 2 MB | operand tiles and streams, no error |

Table: The per-operation validator limits a model must satisfy, with each limit and the result of exceeding it. {#tbl:c17-design-rules}

## Reference: the per-operation validator envelopes

The shape rules are enforced one operation at a time by a family of per-layer validators, the `_ANECValidate<Op>Layer` checkers the compiler runs as the back-end legalizer.
There are 50 per-layer validators, and the same code runs both when a high-level model is segmented and when a hand-authored layer is compiled, so a validator that accepts a shape never drifts from the real compile outcome.
[Table](#tbl:c17-validators) gives the binding constraint and reject string for the operations a model presents most often, read from the `_ANECValidate<Op>Layer` family.

| Operation | Validator | Binding constraint | Reject string |
| --- | --- | --- | --- |
| Convolution | `Conv` | kernel W, H, D each within the per-chip `[min, max]`; in-C and out-C each divisible by groups; groups within the HAL group range | `"Invalid conv kernel %s = %zd, It should be in [%zd,%zd]"`; `"input/output channels should be divisible by num group"` |
| Linear | `Linear` | exactly one input; input rank under 5 | `"Linear layer must have only one single input."`; `"ANE cannot support Linear with input rank >= 5"` |
| Matrix multiply | `MatrixMult` | exactly two inputs; depth one on both operands; output channel equals the left channel; left width plus padding equals right channel; output channel bytes fit the kernel-memory budget | `"Matrix mult. layer can only have two bottoms"`; `"depth > 1 is not supported for MatMult"`; `"can not fit the Kmem"` |
| Pooling | `Pool` | one input; window per axis under the input extent; padding under the kernel; max-pool padding negative, min-pool positive | `"Pool layer must have only one single input."`; `"Pooling mode \"%s\" is not available on this ANE architecture."` |
| Reduction | `Reduction` | one input; each axis at most 4; non-reduced output extent equals input extent, reduced output extent one; square-after-reduce requires the A14 family flag | `"Reduction layer can only have one bottom"`; `"square operation after reduction is not supported"` |
| Arg-min/arg-max | `ArgMinMax` | channel reduction at most 2048, otherwise height or width at most 2048; padding non-negative and under the kernel; equal left and right padding; pool stride in {1, 2, 4} | `"ArgMinMax left padding value should be smaller than kernel width %d, but %d is given"` |
| Layer norm | `LayerNorm` | one input; output type float; channel divisible by the group count; grouped form requires depth one | `"... does not yet support depth > 1"` |
| Softmax | `Softmax` | one input; non-empty axis set; output type float; the general full-axis form is family-gated | `"Softmax is not supported by this ANE architecture"` |
| Transpose | `Transpose` | one input; each dimension appears once; extent at most 16384 on A13 through A15 and 65536 on A16; without three-dimensional support a channel transpose must factor into {2, 3, 4, 8} with height one | `"NE Input Transpose is not supported for this arch"` |
| Concat | `Concat` | at least two inputs; every input matches the first on every non-concatenated axis; constant positive axis; matching layout | `"ANE Concat supports only supports const positive axis"`; `"both concat inputs must have the same layout"` |
| Pad | `Pad` | one input; height or width axes only, no channel or depth padding; reflect and symmetric modes require the texture engine | `"Channel padding is not supported on ANE"`; `"Architecture does not support padding mode."` |
| Broadcast | `Broadcast` | one input; broadcast only from a length-one axis; depth-axis broadcast requires a family flag absent on the M1 | `"Broadcast along depth axis is not supported on this architecture"` |
| Gather | `Gather` | index channel divisible by the interleave factor; on the M1 the software envelope requires data batch one, data depth one, index channel three, index width one, index depth one, and a gather-axes count of three | `"Cannot decompose layer on this architecture"` |
| Reshape | `Reshape` | one input; element count preserved; rank at most 5 | `"Cannot reshape a tensor of rank > 5"` |

Table: Per-operation validator constraints and reject strings, read from the `_ANECValidate<Op>Layer` family. {#tbl:c17-validators}

A validator that accepts a shape does not guarantee the operation compiles.
A second class of operations passes the schema validator and then fails the code generator below it, the attested-is-not-reachable split of chapter 4.
On the M1 the sort and dynamic-slice validators accept their inputs and the code generator rejects the lowering, and top-k compiles only outside a specific forbidden parameter band.
The validator predicts schema reachability; only a compile-and-run on the target confirms an operation runs.

## Reference: the channel and divisibility rules

[Table](#tbl:c17-divisibility) lists the layout and divisibility rules that separate a fast layer from a padded one, with the constraint and the consequence of missing each.

| Rule | Constraint | Consequence |
| --- | --- | --- |
| Channel interleave | channel a multiple of the family interleave factor, found from the HAL table and the dimensions | a non-multiple pads out to one, leaving lanes unused |
| Width granule | last-axis width aligned to the 16-byte direct-memory-access granule, the times-16 quantum | misaligned width pads to the granule |
| Group-to-core ratio | active neural-engine count divisible by the group count, four cores on the M1 so groups in {1, 2, 4} | a non-dividing count loses the one-to-one core-to-lookup-table mapping and runs slower |
| Max-pool channel | input channel under the per-operation maximum-pool channel bound | a large-channel max-pool tiles |
| Unicast channel | channel at most the unicast input-channel maximum for the broadcast-to-all-cores path | above it the operation takes the multicast lowering |

Table: The layout and divisibility rules that separate a fast layer from a padded one, with the constraint and the consequence of missing it. {#tbl:c17-divisibility}

Depthwise convolution, where the group count equals the input channel count, lowers to the channelwise path and is exempt from the group-to-core preference.
