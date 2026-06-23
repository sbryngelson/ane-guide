# 13. Vision, convolution, and encoders

> A 256-channel 3x3 convolution runs about 3.8 times faster than the GPU at about 9 times the energy efficiency, and the engine draws less absolute power on every workload class measured.
> Serve encoders and embeddings on the engine below a batch of about 23, a self-attention block below a batch of about 6, and vision convolution at every batch.
> On-engine image preprocessing arrives on the A14 generation and later; the M1 has no texture engine.
> On the M1 and the A14, keep slice and crop source magnitudes below 4094 on the width axis or they saturate to infinity.

Convolution and vision are the work the engine is built for.
This chapter gives the measured economics of that work against the GPU, and the batch threshold below which encoder serving stays on the engine.
It also covers the on-engine image preprocessing path and the family that gates it, and the model-construction rules that keep a vision or encoder model on the engine end to end.

## Convolution datapath

A 3x3 convolution at 256 channels runs about 3.8 times faster on the engine than the same kernel on the GPU, at about 9 times the energy efficiency per result [AppleCoreML].
The advantage widens with depth.
A sixteen-deep stack of 3x3 convolutions at 256 channels runs about 2 times faster than the GPU and at about 14.5 times its energy efficiency on the M1, the largest power difference in the M1 workload set.
On the M5 it runs about 4.2 times faster at about 13 times the efficiency.
A ResNet-18 forward pass runs about 6.1 times faster than the GPU reference and at about 11 times the energy per inference.

Two properties of the datapath produce this.
The first is Winograd.
Any dense 3x3 stride-1 convolution with enough channels lowers to the F(2x2, 3x3) transform, which replaces 36 direct multiplies per 2x2 output tile with 16, a factor of about 2.25 reduction in multiplies.
The compiler selects this path automatically for dense non-unicast 3x3 stride-1 layers and falls back to direct convolution otherwise, so a model written with 3x3 stride-1 layers gets the reduction without any annotation.
The conv-relevant consequence is that a layer takes Winograd only with enough channels to amortize the transform, and a float kernel requires more than a non-float one; the full eligibility test, two tile sizes, and work-threshold derivation are in chapter 20.
There is no accumulator widening tied to the transform; its precision safety is that higher float work threshold, not a wider accumulator.
The second property is the engine's lower power draw.
On a large compute-bound matrix multiply the engine draws 4.4 W against the GPU's 32.5 W, a 4.0 times efficiency advantage that holds even on the matrix-multiply class where the GPU runs faster.
Across the M1 workload set the engine runs at 2 to 14.5 times the GPU's energy efficiency, drawing less absolute power on every class measured.

The end-to-end convolution ceiling on the M1 is about 1.8 fp16 TFLOP/s at about 1.78 W, or about 643 GFLOP/s per watt under load.
A 3x3 convolution at 256 channels and a 28 by 28 feature map reaches that peak: it runs in about 0.51 ms at about 1823 GFLOP/s.
Throughput rises with spatial size as the fixed per-eval overhead amortizes, and the same 3x3 64-channel kernel runs at about 63 GFLOP/s at 16 by 16 and about 1102 GFLOP/s at 128 by 128.

## MAC array and its tiling

The engine is a fixed-geometry multiply array supplied by DMA engines that re-base per tile.
On the M1 the array is four NE cores, read from the per-chip parameter table as a core count of 4, with each core a two-dimensional multiply tile backed by an accumulator file of 8 work-units.
The compiler assigns output channels across the cores by a strided round-robin: channel $c$ is on core $c \bmod 4$, so the four cores are independent parallel slices and the scheduling granule is one output channel.

The parallelism is measured directly.
Driving a single heavy convolution at each output-channel count in a back-to-back dispatch loop, where the per-dispatch overhead pins the dispatch rate constant, the sustained multiply rate rises in exact integer multiples with the number of cores active, as [table](#tbl:c13-core-scaling) records at one through four active cores.

| Output channels | Cores active ($c \bmod 4$) | Engine power, net | GMAC/s | Ratio to one core |
| ---: | :---: | ---: | ---: | ---: |
| 1 | 1 | 811 mW | 3.8 | 1.00x |
| 2 | 2 | 822 mW | 7.6 | 2.00x |
| 3 | 3 | 831 mW | 11.4 | 3.00x |
| 4 | 4 | 843 mW | 15.4 | 4.05x |

Table: Per-core throughput and power scaling on the M1, measured at a fixed dispatch rate. {#tbl:c13-core-scaling}

The power rail steps about 10 to 11 mW per added core over an always-on floor near 800 mW, the floor being the base plus dispatch domain and each step one of the four independently power-gated compute domains turning on.
Above four output channels the array fills more lanes inside the same four cores, and the rate keeps rising linearly while the dispatch rate stays floor-bound.

The compiler tiles output channels into output-channel groups sized to the accumulator file.
The group size is about the 8-accumulator budget divided by the kernel-element count $k_w k_h k_d$, rounded down to a power of two and capped by a per-element byte budget of 32, 16, or 8 bytes depending on the weight format.
A 3x3 convolution thus has about 9 times the per-channel accumulator pressure of a 1x1, so its group is about 9 times smaller and it needs more passes.
The measured signature is a super-linear cost step: a 1x1 fp16 convolution doubles its per-layer cost once the output channel count crosses the accumulator file between about 192 and 256 channels.
A 3x3 convolution reaches that pass-doubling threshold at fewer channels for the same reason.
This is the mechanism that makes Winograd worth selecting for 3x3: cutting the effective kernel-element count enlarges the group and relieves the accumulator pressure the threshold makes visible.

## Convolution variants and their lowering

The convolution variants all map onto the same multiply array, and what differs is how the compiler tiles the weights and sets up the channel grouping, which [table](#tbl:c13-conv-variants) gives variant by variant.

| Variant | Lowering onto the datapath |
| --- | --- |
| Standard 2D convolution | native; weights tiled into output-channel groups, channels assigned across the cores by a strided round-robin |
| Transpose or deconvolution | native; runs as a fractionally-strided forward convolution on the same array, and serves the convolution data gradient |
| Dilated | lowered by a space-to-batch decomposition with a factor list of 2, 3, 4, or 8, folding the dilation into a strided input gather so the kernel itself stays dense |
| Depthwise | native; each channel is its own group with no cross-channel reduction, one input channel per output channel |
| Grouped | native; the group count partitions the channel-to-core assignment, and runs best when the group count divides the channel count and the core count |
| 3D convolution | capability present in the hardware parameter table but not reachable on the direct path; the compiler reports the operation as not implemented on every backend |

Table: How each convolution variant lowers onto the multiply array on the M1. {#tbl:c13-conv-variants}

The speed and efficiency advantage holds across the workload classes the engine is built for, which [table](#tbl:c13-engine-vs-gpu) gives from the single convolution through ResNet to the encoder.

| Workload | Engine vs GPU speed | Engine vs GPU efficiency |
| --- | --- | --- |
| 3x3 convolution (256 channels) | 3.8x faster | 9x more efficient |
| Convolution stack, 16 deep at 256 channels (M1) | 2x faster | 14.5x more efficient |
| Convolution stack, 16 deep at 256 channels (M5) | 4.2x faster | 13x more efficient |
| ResNet-18 forward | 6.1x faster | 11x more efficient |
| Batched matrix multiply | faster below N of 2048 | more efficient at every batch size |
| Single-sentence encoder | 4.4x faster | faster at low to moderate batch |

Table: Engine versus GPU speed and efficiency across convolution, ResNet, matrix multiply, and encoder workloads. {#tbl:c13-engine-vs-gpu}

## Encoders and embeddings

An encoder forward pass favors the engine at low to moderate batch on both latency and energy.
A single-sentence encoder runs about 4.4 times faster than the GPU.
A twelve-layer encoder forward runs about 1.5 times faster and at about 18 times the energy per inference.
Short-sequence attention, a transformer block at sequence length 197, is at once the fastest, most efficient, and most accurate in fp16 on the engine, because its partial sums stay in range and the wide accumulator holds their precision.

Batch size moves the choice, because serving batches requests.
The engine saturates near a batch of 1 while the GPU scales with batch, which produces a throughput crossover.
On a true-batched encoder block the GPU overtakes the engine on throughput near a batch of 23, and on a self-attention block near a batch of 6.
The energy crossover is at a larger batch than the throughput crossover.
On vision convolution serving the energy crossover never appears: the engine leads throughput by 3.6 to 5.7 times and energy by 6 to 10 times at every batch from 1 to 256.

Serve encoders and embeddings on the engine below a batch of about 23, and a self-attention block below a batch of about 6.
Above those points the GPU is the throughput device.
Vision convolution serving stays on the engine at every batch.

## On-engine image preprocessing

Resize, crop-and-resize, grid sample, affine warp, and reflective or symmetric padding run on a single hardware sampling datapath, the texture engine, on the A14 generation and later [AppleVision].
The datapath is a DMA-side sampler fused onto a layer rather than a separate pass: it reads a coordinate or box tensor through a fixed interleave and applies bilinear or nearest-neighbor interpolation in line with the convolution that consumes its output.
The payoff is the elimination of a host preprocessing stage.
Image dequantization and resize execute on the engine in the same program as the model, so the resampled tensor never makes a round trip to the host between the camera frame and the first convolution.

One datapath backs a fixed set of front-end operations, each gated by the same parameter-table bool, which [table](#tbl:c13-texture-ops) lists with how each lowers on the A14 generation and later.

| Operation | Bottoms | Lowering on the A14 and later |
| --- | ---: | --- |
| Resize, upsample | 1 | native texture-engine resize unit |
| Crop-and-resize, ROI-align | 2 | native; a denormalization scale-and-bias pair precedes the index input, box tensor in fp16 |
| Resample, grid-sample, warp | 2 | native; coordinate tensor of 1 or 2 channels read through the index interleave |
| Affine transform | 2 | decomposes to resample with a computed coordinate grid; matrix fp16, six coefficients |
| Resize-as | 2 | decomposes to resize |
| Reflective or symmetric padding | 1 | uses the texture-engine pad mode |
| Hardware gather | 2 | uses the texture-engine index interleave path |

Table: The texture-engine operation set and how each lowers on the A14 generation and later. {#tbl:c13-texture-ops}

The sampler reads its coordinate or box tensor through an interleave of 1, 2, 3, 4, or 8, set by the box-coordinate layout.
A two-corner box such as `Y0X0Y1X1` selects interleave 4, an origin-and-size box selects 4, a two-coordinate point selects 2, and a full four-coordinate batched box selects the default 8.
The sampling method is linear, that is bilinear, or nearest-neighbor, and the interpolation weights and the crop scale program two register arrays at the DMA-side sampler.
On the M1 the whole set decomposes or rejects.
Resize becomes a channelwise deconvolution for integer upsample or a transpose followed by a convolution otherwise, affine warp rejects outright with a not-supported-on-this-architecture diagnostic, and symmetric padding rejects.
Gather routes through a limited software envelope that requires a gather-axis size of 3 and a batch and depth of 1.

A single bool in the per-chip parameter table gates the capability, and that bool is zero on the M1.
The gate is one bit with no finer per-operation granularity: crop-and-resize, resample, affine, native resize, the hardware gather index path, and reflective or symmetric padding all turn on together at the A14.

The same gate has a precision caveat on the M1 and the A14.
The crop and slice path applies a width-axis gain of 16, so a source value above 4094 on the width axis saturates to plus or minus infinity, while height, channel, and batch offsets stay free of the saturation.
Chapter 3 derives the bound and chapter 19 gives the build-time guard.
The preprocessing consequence is to keep slice and crop source magnitudes below 4094 on the width axis on the M1 and the A14.
The A15 generation and later sampler avoids this saturation.

## Keeping the model on the engine

The datapath runs best with a small set of model-construction choices.

- Prefer 3x3 stride-1 dense convolutions over 5x5 or strided forms where the
  model allows, so the Winograd path is taken and each layer gets the factor of
  about 2.25 multiply reduction.
- Keep the largest single per-layer operand at or below the 2 MB on-chip working
  set. A convolution whose live tiles exceed 2 MB is tiled and streamed from
  DRAM, which adds transfer traffic and moves the layer off the compute ceiling.
- Fuse the whole graph into one program so the about 0.23 ms per-eval dispatch
  floor on the M1 is paid once rather than per layer. A fixed-iteration stencil
  fused as one graph amortizes that floor across every step.
- Fold normalization and activation into the convolution. Per-output-channel
  scale and bias and a fused nonlinearity run on the convolution output at no additional cost,
  so a batch-normalization or a clamped activation after a convolution adds no
  separate dispatch.
- Serve encoders below the batch threshold of the previous section, and keep
  preprocessing on the engine only on the A14 and later, where the texture
  engine is present.

The native convolution backend operation holds its geometry in a fixed set of attributes that the compiler reads to tile the layer onto the multiply array, given in [listing](#lst:c13-conv-backend).

```mlir
anec.convolution(%input, %weights, %bias) {
  strides         = [sh, sw],          // 1 keeps the Winograd path for a 3x3 kernel
  explicit_padding = [pt, pb, pl, pr], // or padding_style
  groups          = g,                 // 1 standard; g == Cin == Cout is depthwise
  dilation_rates  = [dh, dw],          // folded to a strided input gather, up to 8
  kernel_sizes    = [kh, kw],          // fp16 kernel width bounded at 13 on the M1
  weights_layout                        // weights are output-channel-major, [Cout, D, Cin, H, W]
}  // filter and input rank 4 or 5
```

Listing: The native convolution backend operation and the geometry attributes the compiler reads to tile a layer onto the multiply array. {#lst:c13-conv-backend}

The matrix-multiply backend operation that the encoder paths and the convolution weight gradient reduce to holds its contraction direction in two attributes rather than a fixed operand order, as [listing](#lst:c13-matmul-backend) shows.

```mlir
anec.matmul(%lhs, %rhs) { transpose_lhs, transpose_rhs }  // depth D must be 1 on both operands
```

Listing: The matrix-multiply backend operation, which holds its contraction direction in two transpose attributes rather than a fixed operand order. {#lst:c13-matmul-backend}

A trainable convolution lowers its forward and its data gradient onto the native convolution and deconvolution operations, and its weight gradient through an image-to-column expansion into this matrix multiply, because the hardware cross-correlation operation is single-channel only and cannot serve the multi-channel weight gradient directly.

## Fitting a convolution or encoder block on the engine

A vision or encoder block stays on the engine when its layers take the Winograd path, its working set fits on chip, and the whole graph compiles as one program.
The procedure of [listing](#lst:c13-fit-block) builds the convolution stack as one graph, then locates it against the roofline before any device is available.

```python
# Fit a convolution or encoder block on the engine as one fused program.
# Each stage is convolution, then normalization, then activation.

build graph G:
    input x : [1, 256, 28, 28] fp16

    # Stage 1: keep the convolution 3x3 stride-1 so it takes the Winograd path.
    h = conv(x, weights = w1, kernel = [3, 3], stride = 1)
    h = normalize(h)                       # batch or layer norm, folded into the conv output
    h = activation(h)                      # e.g. relu or gelu, on the same output pass

    # Stage 2: another 3x3 stride-1 stage, same fused shape.
    h = conv(h, weights = w2, kernel = [3, 3], stride = 1)
    h = normalize(h)
    h = activation(h)

    output h

# 1. Check the working set stays on chip before committing to the engine.
working_set = max over stages of resident_bytes(stage)   # in MB
if working_set > 2.0:
    # Too large: it would tile and stream from DRAM. Shrink the operand
    # (fewer channels or smaller tile) and re-check before any other tuning.
    reshape operands until working_set <= 2.0

# 2. Fuse the whole block into ONE program so the per-call floor is paid once.
program = fuse_and_compile(G, target = H13)   # one dispatch, the 0.23 ms floor paid a single time

# 3. Run the fused program on an input image.
output = run(program, x = image)

# Note: input preprocessing can join the same graph only on A14 and later,
# where the texture engine is present.
return output
```

Listing: Fitting a convolution or encoder block on the engine as one fused program. {#lst:c13-fit-block}

## Reference: convolution and encoder economics on the M1

[Table](#tbl:c13-reference) collects the M1 convolution and encoder constants this chapter measures, from the GPU speed ratios through the datapath geometry to the batch thresholds.

| Constant | M1/H13 value |
| --- | ---: |
| 3x3 convolution (256 channels) speed versus GPU | 3.8x faster |
| 3x3 convolution (256 channels) efficiency versus GPU | 9x more efficient |
| Convolution-stack speed, 16 deep at 256 channels | 2x faster |
| Convolution-stack efficiency, 16 deep at 256 channels | 14.5x more efficient |
| Matrix-multiply power, engine versus GPU | 4.4 W versus 32.5 W |
| Energy-efficiency range across the workload set | 2x to 14.5x |
| End-to-end convolution ceiling | 1.8 fp16 TFLOP/s at 1.78 W |
| Convolution efficiency under load | 643 GFLOP/s per watt |
| 3x3 256-channel 28 by 28 peak | 0.51 ms at 1823 GFLOP/s |
| Spatial throughput, 3x3 64-channel | 63 GFLOP/s at 16 by 16, 1102 GFLOP/s at 128 by 128 |
| Per-eval dispatch floor | 0.23 ms |
| On-chip working-set threshold | 2 MB |
| NE-core count | 4 |
| Accumulator file per core | 8 work-units |
| Output-channels per cycle, fp16 default / int8 fast path | 4 / 8 |
| Channel-to-core assignment | strided round-robin, channel $c$ on core $c \bmod 4$ |
| Output-channel-group pass-doubling threshold, 1x1 fp16 | 192 to 256 channels |
| Per-core power step over the dispatch floor | about 10 mW |
| Winograd work threshold, non-float / float / packed | 8 / 16 / 32 |
| Texture-engine interleave factors | 1, 2, 3, 4, 8 |
| Encoder-serving batch threshold | 23 |
| Self-attention-serving batch threshold | 6 |
| Q.4 crop-scale saturation, width axis | 4094, where $4094 \times 16 = 65504$ |

Table: The M1 convolution and encoder constants for vision and encoder workloads. {#tbl:c13-reference}
