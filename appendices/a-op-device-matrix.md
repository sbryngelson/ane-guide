# Appendix A. Operation-by-device matrix

> This appendix is the full per-operation, per-family status reference behind chapter 4.
> Read down a family column to see what compiles and runs on that chip, and read the status marks and note for the gate and the route.

Each row is one intermediate-language operation, grouped by operation class, with its status on each Mac engine family from the M1 through the M5.
Chapter 4 summarizes this table; the cells here are the reference.

The status marks are fixed:

- Native: the operation compiles and runs on that family on the direct engine path.
- Family-gated: no path on the listed family, native from the family named in the note.
- Bridge: reachable only through a decompose, software fallback, or compiler-internal route, never as a standalone code-generated operation.
- No path: rejected on every family from the M1 through the M5, computed off-engine.

The family columns are M1 (H13, A13), M2 (H14, A14), M3 (H15, A15), and M4 and M5 (H16 and H17s, A16 and A17).
The A11 and A12 engines are below the floor that runs any of this vocabulary and are out of scope for the table.

The M1, M2, and M5 columns are measured on physical silicon.
The M3 column and the M4 part of the merged M4 and M5 column are decompile-derived predictions from the per-chip tables, so a per-cell status there is a predicted capability rather than a measured one.

The table covers the 187 intermediate-language operations the compiler exposes.
Of these, about 108 are native on the M1: the full elementwise, compare, activation, convolution, pooling, structural, and quantization vocabulary, plus the reduction, normalization, softmax, square-root family, fused attention, tile, and space-channel set.
Nine need the M2 or later: the texture-engine operations (crop-resize, resample, affine, hardware gather) and the rank and sort bridge (top-k, sort, dynamic slice).
Four need the M3 or later: native `sin` and `cos`, the hardware random generator, and the whole-tensor argument reductions on the intermediate-language route.
Thirty-seven are rejected on every family and decompose on the host.
About twenty-four are compiler-internal: mapped but with no observed standalone code generation, reachable only inside a wrapping construct.

## Per-chip numeric limits

The status of an operation is one axis; the numeric envelope it runs in is the other.
[Table](#tbl:apa-numeric-limits) gives that envelope across the five capability tiers, measured from the live compiler by calling every per-architecture parameter constructor on a single M1, and a dash marks an unsupported value.
The `older` column is the pre-A13 legacy targets the compiler still has parameter tables for, below the floor of the operation-status table above.

| Limit | older | M1, A13 | A14 | A15 | A16, M5 |
| --- | --- | --- | --- | --- | --- |
| max kernel W (default format, large) | 29 | 29 | 32 | 32 | 32 |
| max kernel W (fp16, large) | 13 | 13 | 16 | 16 | 16 |
| max kernel W (default, small) | 15 | 15 | 16 | 16 | 16 |
| max kernel W (fp16, small) | 7 | 7 | 8 | 8 | 8 |
| min kernel W (default / fp16, large) | 16 / 8 | 16 / 8 | 1 / 1 | 1 / 1 | 1 / 1 |
| max kernel H (large / small) | 29 / 15 | 29 / 15 | 32 / 16 | 32 / 16 | 32 / 16 |
| max kernel D (large / small) | 1 / 1, no 3D | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 |
| max patch W / H / D | 15 / 15 / 0 | 28 / 28 / 15 | 31 / 31 / 15 | 31 / 31 / 15 | 31 / 31 / 15 |
| max tensor W / H | 16384 | 16384 | 16384 | 16384 | 65536 |
| max tensor D | 1 | 16384 | 16384 | 16384 | 65536 |
| max tensor C | 65536 | 65536 | 65536 | 65536 | 65536 |
| max tensor N (batch) | 4096 | 65536 | 65536 | 65536 | 65536 |
| max transpose W / H | 0, always split | 16384 | 16384 | 16384 | 65536 |
| reduction-to-transpose threshold | none | 192 | 192 | 384 | 384 |
| group-conv decompose limit (Cin·kW·kH) | 64 | 2048 | 2048 | 2048 | 2048 |
| stride factor list | [2,3,4,8] | [2,3,4,8] | [2,3,4,8] | [2,3,4,8] | [2,3,4,8] |
| matmul SRAM working set | 2 MB, M9 1 MB | 2 MB | 2 MB | 2 MB | 2 MB |
| DMA width granule | 16 B | 16 B | 16 B | 16 B | 16 B |
| patch-width floor / max | none | 16 / 512 px | 16 / 512 | 16 / 512 | 16 / 512 |
| instruction alignment | 256 B | 256 B | 16 B | 16 B | 16 B |
| has texture engine | no | no | yes | yes | yes |
| kernel-memory budget | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |
| activation-LUT budget | 150 B | 86 B | 86 B | 86 B | 86 B |
| context-switch live-tensor limit | 2 | 2 | ∞ | ∞ | ∞ |

Table: The per-chip numeric limits of the engine, measured from the live compiler across the five capability tiers. {#tbl:apa-numeric-limits}

The four generational dividing lines are visible in this table.
The M1 adds the depth and three-dimensional axis and every reduction-class operation.
The A14 adds the texture engine.
The A15 raises the reduction-to-transpose threshold from 192 to 384 and adds native trigonometry.
The A16 quadruples the maximum tensor and transpose dimensions from 16384 to 65536.

## Convolution, matrix multiply, and pooling

[Table](#tbl:apa-conv-matmul-pool) lists the convolution, matrix-multiply, and pooling operations with their per-family status and the lowering note for each.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `conv` | Native | Native | Native | Native | M1 kernels up to 29x29, M5 up to 32x32; Winograd auto-selected for eligible 3x3 stride-1 convs |
| `conv_transpose` | Native | Native | Native | Native | Deconvolution; strided axes use the small-kernel caps |
| `linear` | Native | Native | Native | Native | Folds to convolution when the right operand fits the on-chip working set |
| `linear_activation` | Native | Native | Native | Native | Fused linear and activation |
| `matmul` | Native | Native | Native | Native | Engine lane or convolution fold; same tensor caps as convolution |
| `ne_matmul` | Native | Native | Native | Native | Private engine-lane matrix-multiply unit |
| `einsum` | Native | Native | Native | Native | Lowers to a matmul and transpose chain |
| `ne_conv` | Native | Native | Native | Native | Private engine-lane convolution unit |
| `avg_pool` | Native | Native | Native | Native | Window up to 29 on the M1, up to 31 from the M2 |
| `max_pool` | Native | Native | Native | Native | |
| `l2_pool` | Native | Native | Native | Native | Lookup-table pool |
| `ne_pool` | Native | Native | Native | Native | Private engine-lane pooling unit |
| `pe_pool` | Native | Native | Native | Native | Private planar-engine pooling unit |
| `pe_elementwise` | Native | Native | Native | Native | Private planar-engine elementwise unit |
| `pe_goc` | Bridge | Bridge | Bridge | Bridge | Private planar-engine gain-offset unit, compiler-internal |
| `ne_bypass` | Bridge | Bridge | Bridge | Bridge | Private engine-lane bypass unit, compiler-internal |
| `scaled_dot_product_attention` | Native | Native | Native | Native | Runs on the matmul and softmax path, not texture-gated |

Table: Convolution, matrix-multiply, and pooling operations by device family. {#tbl:apa-conv-matmul-pool}

The `ne_` and `pe_` rows are private engine-lane and planar-engine unit selections of the same convolution, matrix-multiply, pooling, and elementwise atoms, not separate operations.

## Normalization

[Table](#tbl:apa-norm) gives the normalization operations, native on every family from the M1.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `batch_norm` | Native | Native | Native | Native | Inference fold-to-affine; native statistics form from the M1 |
| `layer_norm` | Native | Native | Native | Native | |
| `instance_norm` | Native | Native | Native | Native | |
| `l2_norm` | Native | Native | Native | Native | |
| `local_response_norm` | Native | Native | Native | Native | Measured on the M1 |

Table: Normalization operations by device family. {#tbl:apa-norm}

## Elementwise arithmetic

[Table](#tbl:apa-elementwise) gives the elementwise arithmetic operations, where only `mod` takes no engine path.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `abs` | Native | Native | Native | Native | |
| `add` | Native | Native | Native | Native | Constant and tensor forms |
| `sub` | Native | Native | Native | Native | Lowered to add of a negated constant |
| `mul` | Native | Native | Native | Native | Constant and tensor forms |
| `real_div` | Native | Native | Native | Native | General divide |
| `floor_div` | Native | Native | Native | Native | Lookup-table assisted |
| `pow` | Native | Native | Native | Native | |
| `square` | Native | Native | Native | Native | |
| `sqrt` | Native | Native | Native | Native | Lookup-table activation |
| `rsqrt` | Native | Native | Native | Native | Lookup-table |
| `inverse` | Native | Native | Native | Native | Reciprocal lookup-table |
| `maximum` | Native | Native | Native | Native | |
| `minimum` | Native | Native | Native | Native | |
| `mod` | No path | No path | No path | No path | Decompose on host |
| `cumsum` | Native | Native | Native | Native | Native through a curated runtime path, not the standard compile path; M1 measured |

Table: Elementwise arithmetic operations by device family. {#tbl:apa-elementwise}

## Comparison and logical

[Table](#tbl:apa-compare-logical) gives the comparison and logical operations, the bitwise-logical ones decomposing on the host.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `equal` | Native | Native | Native | Native | |
| `not_equal` | Native | Native | Native | Native | |
| `greater` | Native | Native | Native | Native | |
| `greater_equal` | Native | Native | Native | Native | |
| `less` | Native | Native | Native | Native | |
| `less_equal` | Native | Native | Native | Native | |
| `logical_not` | Native | Native | Native | Native | |
| `select` | Native | Native | Native | Native | The where operation |
| `logical_and` | No path | No path | No path | No path | Decompose through minimum or multiply on host |
| `logical_or` | No path | No path | No path | No path | Decompose through maximum on host |
| `logical_xor` | No path | No path | No path | No path | Decompose through not-equal on host |

Table: Comparison and logical operations by device family. {#tbl:apa-compare-logical}

## Activations

[Table](#tbl:apa-activations) gives the activation operations, native on every family and most lookup-table backed.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `relu` | Native | Native | Native | Native | |
| `relu6` | Native | Native | Native | Native | Lookup-table |
| `leaky_relu` | Native | Native | Native | Native | Lookup-table |
| `prelu` | Native | Native | Native | Native | Per-channel slope; native at rank 3 or above |
| `clamped_relu` | Native | Native | Native | Native | Lookup-table |
| `thresholded_relu` | Native | Native | Native | Native | Lookup-table |
| `threshold` | Native | Native | Native | Native | Lookup-table |
| `clip` | Native | Native | Native | Native | The clamp operation |
| `elu` | Native | Native | Native | Native | Lookup-table |
| `sigmoid` | Native | Native | Native | Native | Includes the hard variant |
| `sigmoid_hard` | Native | Native | Native | Native | Lookup-table |
| `tanh` | Native | Native | Native | Native | Lookup-table |
| `scaled_tanh` | Native | Native | Native | Native | Lookup-table |
| `gelu` | Native | Native | Native | Native | Lookup-table approximation |
| `silu` | Native | Native | Native | Native | Also named swish; lookup-table |
| `softmax` | Native | Native | Native | Native | Lookup-table |
| `softplus` | Native | Native | Native | Native | Lookup-table |
| `softplus_parametric` | Native | Native | Native | Native | Lookup-table |
| `softsign` | Native | Native | Native | Native | Lookup-table |
| `erf` | Native | Native | Native | Native | Lookup-table |
| `exp` | Native | Native | Native | Native | Lookup-table |
| `exp2` | Native | Native | Native | Native | Lookup-table |
| `log` | Native | Native | Native | Native | Lookup-table |
| `sign` | Native | Native | Native | Native | Lookup-table |
| `ceil` | Native | Native | Native | Native | Lookup-table |
| `floor` | Native | Native | Native | Native | Lookup-table |
| `round` | Native | Native | Native | Native | Round-to-nearest lookup-table |

Table: Activation operations by device family. {#tbl:apa-activations}

## Reduction

[Table](#tbl:apa-reduction) gives the reduction operations, where `reduce_argmin` is gated and `reduce_prod` takes no path.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `reduce_sum` | Native | Native | Native | Native | Reduced axis at or above 192 takes the transpose route, at or above 384 from the M3 |
| `reduce_mean` | Native | Native | Native | Native | |
| `reduce_max` | Native | Native | Native | Native | |
| `reduce_min` | Native | Native | Native | Native | |
| `reduce_sum_square` | Native | Native | Native | Native | The reduce-then-square fusion is M2 onward; the M1 emits an extra fp16 round |
| `reduce_l1_norm` | Native | Native | Native | Native | |
| `reduce_l2_norm` | Native | Native | Native | Native | |
| `reduce_log_sum` | Native | Native | Native | Native | Lookup-table assisted |
| `reduce_log_sum_exp` | Native | Native | Native | Native | Lookup-table assisted |
| `reduce_argmax` | Native | Native | Native | Native | Per-axis argmax on all families |
| `reduce_argmin` | Bridge | Bridge | Native | Native | Per-axis argmin; the intermediate-language route is gated to the M3, the bridge route works on the M1 and M2 |
| `reduce_prod` | No path | No path | No path | No path | Decompose through log-sum-exp on host |

Table: Reduction operations by device family. {#tbl:apa-reduction}

The whole-tensor argument reductions `global_argmax` and `global_argmin` follow the same gate as `reduce_argmin`: native on the intermediate-language route from the M3, reachable through the bridge on the M1.

## Data movement and structural

[Table](#tbl:apa-data-movement) gives the data-movement and structural operations, the largest class, spanning reshape, slice, gather, scatter, and the space-channel set.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `reshape` | Native | Native | Native | Native | Metadata edit |
| `reshape_like` | Native | Native | Native | Native | |
| `expand_dims` | Native | Native | Native | Native | |
| `squeeze` | Native | Native | Native | Native | |
| `flatten2d` | Native | Native | Native | Native | |
| `transpose` | Native | Native | Native | Native | Capped by the maximum transpose extent, 16384 through the M3, 65536 on the M5 |
| `concat` | Native | Native | Native | Native | DMA |
| `split` | Native | Native | Native | Native | |
| `stack` | Native | Native | Native | Native | |
| `pad` | Native | Native | Native | Native | Constant pad is native everywhere; symmetric and reflect pad are texture-gated, software on the M1 and native from the M2 |
| `slice_by_size` | Native | Native | Native | Native | M1 and M2 nonzero width-offset routes through a fixed-point crop-DMA that saturates a magnitude above 4094 to infinity; clean from the M3 |
| `slice_by_index` | Bridge | Bridge | Bridge | Bridge | Static-offset slice folds into the descriptor inside a graph |
| `slice_update` | Native | Native | Native | Native | |
| `reverse` | Native | Native | Native | Native | Measured on the M1 |
| `reverse_sequence` | No path | No path | No path | No path | Decompose on host |
| `tile` | Native | Native | Native | Native | Factors of 2, 3, 4, and 8 |
| `gather` | Native | Native | Native | Native | M1 software path valid only for a batch of one and a depth of one; the hardware path is M2 onward |
| `gather_along_axis` | Native | Native | Native | Native | Same M1 envelope caveat |
| `gather_nd` | Bridge | Native | Native | Native | M1 software envelope only (batch one, depth one, three-element index channel); native texture path from the M2 |
| `scatter` | No path | No path | No path | No path | Decompose on host |
| `scatter_along_axis` | No path | No path | No path | No path | Decompose on host |
| `scatter_nd` | No path | No path | No path | No path | Decompose on host |
| `depth_to_space` | Native | Native | Native | Native | The pixel-shuffle operation |
| `space_to_depth` | Native | Native | Native | Native | The pixel-unshuffle operation |
| `pixel_shuffle` | Native | Native | Native | Native | Engine-lane reorganization, factors of 2, 3, 4, and 8; z-factor must be 1 |
| `pixel_unshuffle` | Native | Native | Native | Native | Engine-lane reorganization; input dimension divisible by the factor |
| `space_to_batch` | Native | Native | Native | Native | Factor in 2, 3, 4, 8; batch cap 4096 on older families, 65536 on the newer |
| `batch_to_space` | Native | Native | Native | Native | Inverse of the above |
| `identity` | Native | Native | Native | Native | Aliases a cast or no-op |
| `fill` | Native | Native | Native | Native | Constant tensor producer |
| `fill_like` | Native | Native | Native | Native | Constant tensor producer |
| `range_1d` | Bridge | Bridge | Bridge | Bridge | M1 code generation rejects it; host-precompute the constant |
| `crop` | Native | Native | Native | Native | Slice and crop, distinct from the texture crop-resize |
| `band_part` | No path | No path | No path | No path | Mask on host |
| `non_zero` | No path | No path | No path | No path | Data-dependent shape |
| `one_hot` | No path | No path | No path | No path | Decompose through an identity gather on host |
| `shape` | No path | No path | No path | No path | Static-shape graphs only |
| `sliding_windows` | No path | No path | No path | No path | Decompose on host |

Table: Data-movement and structural operations by device family. {#tbl:apa-data-movement}

## Image, resize, and texture

[Table](#tbl:apa-image-texture) gives the image, resize, and texture operations, gated to the texture engine from the A14 with software fallbacks on the M1.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `resize` | Bridge | Native | Native | Native | Texture-gated; M1 takes a software transpose fallback with different rounding, native from the M2 |
| `resize_bilinear` | Bridge | Native | Native | Native | Software fallback on the M1 |
| `resize_nearest_neighbor` | Bridge | Native | Native | Native | Software fallback on the M1 |
| `upsample_bilinear` | Bridge | Native | Native | Native | Software fallback on the M1 |
| `upsample_nearest_neighbor` | Bridge | Native | Native | Native | Software fallback on the M1 |
| `crop_resize` | Family-gated | Native | Native | Native | Texture engine, M2 onward; no host substitution wired |
| `resample` | Family-gated | Native | Native | Native | Texture engine, M2 onward |
| `affine` | Family-gated | Native | Native | Native | Texture engine, M2 onward |
| `pixel_buffer_to_tensor` | Bridge | Bridge | Bridge | Bridge | Four-character-code image input; an entitlement gate, not a chip gate |
| `tensor_to_pixel_buffer` | Bridge | Bridge | Bridge | Bridge | Compiler-internal |
| `gamma` | Bridge | Bridge | Bridge | Bridge | Image-signal operation, compiler-internal |
| `degamma` | Bridge | Bridge | Bridge | Bridge | Image-signal operation, compiler-internal |

Table: Image, resize, and texture operations by device family. {#tbl:apa-image-texture}

## Quantization and dtype

[Table](#tbl:apa-quant-dtype) gives the quantization and dtype operations, with the per-family streaming gates carried in the note column.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `cast` | Native | Native | Native | Native | fp16 to fp32 and bool native on the M1; cast to int32 is rejected on the M1 |
| `quantize` | Native | Native | Native | Native | Not texture-gated |
| `dequantize` | Native | Native | Native | Native | |
| `const` | Bridge | Bridge | Bridge | Bridge | Folded at compile, not a standalone code-generated operation |
| `constexpr_affine_dequantize` | Bridge | Bridge | Bridge | Bridge | int4 lookup-table streams from the M1; int8 and affine fold to fp16 below the M2, and stream from the A14 and M2 |
| `constexpr_lut_to_dense` | Native | Native | Native | Native | Palette and lookup-table stream; int4 lookup-table streams natively from the M1 |
| `constexpr_lut_to_sparse` | Bridge | Bridge | Bridge | Bridge | Folded constant; sparse stream from the M3 |
| `constexpr_blockwise_shift_scale` | Bridge | Bridge | Native | Native | Blockwise stream from the M3; folds to fp16 on the M1 and M2 |
| `constexpr_sparse_blockwise_shift_scale` | Bridge | Bridge | Native | Native | Sparse and blockwise stream from the M3 |
| `constexpr_sparse_to_dense` | Native | Native | Native | Native | Sparse streams natively from the M1 |
| `constexpr_cast` | No path | No path | No path | No path | Rejected on every family |

Table: Quantization and dtype operations by device family. {#tbl:apa-quant-dtype}

## Attention, control flow, and state

[Table](#tbl:apa-attn-control-state) gives the attention, control-flow, and state operations, where the state pair is native and the control-flow operations are compiler-internal.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `read_state` | Native | Native | Native | Native | Stateful; needs the inout tensor-descriptor plumbing for a key-value cache |
| `write_state` | Native | Native | Native | Native | Stateful |
| `tensor_buffer_to_tensor` | Bridge | Bridge | Bridge | Bridge | Ring and streaming buffer mover, reachable inside a stateful graph |
| `tensor_to_tensor_buffer` | Bridge | Bridge | Bridge | Bridge | Compiler-internal |
| `circular_buffer_to_tensor` | Bridge | Bridge | Bridge | Bridge | Ring-buffer reader |
| `tensor_to_circular_buffer` | Bridge | Bridge | Bridge | Bridge | Ring-buffer writer |
| `cond` | Bridge | Bridge | Bridge | Bridge | No standalone code generation; flatten on host |
| `while_loop` | Bridge | Bridge | Bridge | Bridge | No standalone code generation; unroll on host |
| `call` | Bridge | Bridge | Bridge | Bridge | Inlined |

Table: Attention, control-flow, and state operations by device family. {#tbl:apa-attn-control-state}

## Recurrent cells

[Table](#tbl:apa-recurrent) gives the recurrent-cell operations, none of which take an engine path; each unrolls on the host.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `gru` | No path | No path | No path | No path | Unroll to a convolution, matmul, and activation graph on host |
| `lstm` | No path | No path | No path | No path | Unroll on host |
| `rnn` | No path | No path | No path | No path | Unroll on host |

Table: Recurrent-cell operations by device family. {#tbl:apa-recurrent}

## Trigonometric, special, and math

[Table](#tbl:apa-trig-math) gives the trigonometric, special, and math operations, where `sin` and `cos` go native from the M3 and `atan` is the one M1-native primitive.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `sin` | Family-gated | Family-gated | Native | Native | Native from the M3; the M1 and M2 use a host polynomial |
| `cos` | Family-gated | Family-gated | Native | Native | Native from the M3; the M1 and M2 use a host polynomial |
| `atan` | Native | Native | Native | Native | The one trigonometric primitive native on the M1 |
| `tan` | No path | No path | No path | No path | Decompose through a sin and cos identity on host |
| `asin` | No path | No path | No path | No path | Host decomposition |
| `acos` | No path | No path | No path | No path | Host decomposition |
| `atanh` | No path | No path | No path | No path | Host decomposition |
| `asinh` | No path | No path | No path | No path | Host decomposition |
| `acosh` | No path | No path | No path | No path | Host decomposition |
| `sinh` | No path | No path | No path | No path | Host decomposition |
| `cosh` | No path | No path | No path | No path | Host decomposition |
| `cross_product` | Bridge | Bridge | Bridge | Bridge | Reachable through the bridge route, measured on the M1 |
| `cost_volume` | Bridge | Bridge | Bridge | Bridge | Reachable through the bridge route, measured on the M1 |
| `matrix_decomposition` | Bridge | Bridge | Bridge | Bridge | No observed code generation |

Table: Trigonometric, special, and math operations by device family. {#tbl:apa-trig-math}

## Detection and sampling

[Table](#tbl:apa-detection-sampling) gives the detection and sampling operations, the rank and sort bridge gated to the M2 and the random and tensor-list operations off-engine.

| Operation | M1 (A13) | M2 (A14) | M3 (A15) | M4, M5 (A16, A17) | Note |
| ------------------------ | :--------: | :--------: | :--------: | :--------: | ------------------------------ |
| `non_maximum_suppression` | Bridge | Bridge | Bridge | Bridge | Reachable only with a CPU or GPU backend in the mask; the engine-only mask reports not supported on any backend, so it offloads to the CPU or GPU rather than the engine |
| `topk` | Family-gated | Native | Native | Native | Rank and sort bridge, M2 onward; the validator is callable on the M1 but code generation rejects it |
| `argsort` | Family-gated | Native | Native | Native | Sort family, M2 onward; code-generation-rejected on the M1 |
| `random_uniform` | Bridge | Bridge | Native | Native | Hardware generator from the M3; host random below it |
| `random_bernoulli` | No path | No path | No path | No path | Host random |
| `random_categorical` | No path | No path | No path | No path | Host random |
| `random_normal` | No path | No path | No path | No path | Host random |
| `list_gather` | No path | No path | No path | No path | Tensor-list operation |
| `list_length` | No path | No path | No path | No path | Tensor-list operation |
| `list_read` | No path | No path | No path | No path | Tensor-list operation |
| `list_scatter` | No path | No path | No path | No path | Tensor-list operation |
| `list_write` | No path | No path | No path | No path | Tensor-list operation |
| `make_list` | No path | No path | No path | No path | Tensor-list operation |

Table: Detection and sampling operations by device family. {#tbl:apa-detection-sampling}
