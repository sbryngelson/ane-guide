# Appendix B. Hidden-layer catalog

> This appendix is the reference catalog of the hardware-native layer kinds reached by authoring the network description directly, behind chapter 26.

Every row is a native descriptor the conversion path never emits: the descriptor and its `_ANECValidate<Name>Layer` checker are present in the compiler on every target, and the layer is reached by handing the compiler a Unit whose `Type` is the native name and whose `Params` hold the attributes the matching `ZinParse<Name>Unit` parser reads.

## Catalog

[Table](#tbl:apb-catalog) lists each native layer kind with what it computes, its netplist `Type` and compiler symbol, and its family gate.

| Layer kind | What it computes | Netplist `Type` and compiler symbol | Family gate |
| --- | --- | --- | --- |
| Fused attention | $\mathrm{softmax}(QK^{\top}\cdot s + M)\,V$ from four operands (Q, K, V, scale) plus an optional fifth additive mask $M$, with the channel axis holding the sequence | `SDPA`; `ANECSDPALayerDesc`, `ZinParseSDPAUnit`, `anec.sdpa`. The one parsed key is `SubtractMax`, defaulting false in `_ANECSDPALayerDescInitialize` and set true for a correct softmax | All families from the M1 onward; runs on the matmul, softmax, and transpose path, not the texture engine |
| Sort | Full sort along a chosen axis, ascending or descending, values or argsort indices | `Sort`; `ZinParseSortUnit`. Keys: `Direction`, `SortDimension`, `VectorDimension`, `SortIndices`, `Indices` | Validator callable on the M1, but the code generator rejects `Sort` there; runs on later families |
| Top-k | The k largest or smallest along an axis, values or indices, index outputs returned float16-encoded and exact below 2048 | `TopK`; `ZinParseTopKUnit`. Keys: `Type` (Max or Min), `K`, `SortDimension`, `VectorDimension`, `SortIndices`, `Indices` | Runs on the M1 outside a forbidden band: `K` in $\{3, 4\}$ fails the compiler at every width |
| Argument min and max | Spatial or channel argmin and argmax over a kernel window | `ArgMinMax`; `_ANECValidateArgMinMaxLayer`. Keys: `Mode` (SpatialArgMax, ChannelArgMax, SpatialArgMin, ChannelArgMin), `KernelWidth`, `KernelHeight`, `Pad*` | All families from the M1 onward |
| Whole-tensor argument min and max | Argmin or argmax over an entire tensor dimension | `GlobalArgMinMax`. Keys: `Type` (Max or Min), `Dimension` | Gated to the A15 generation; rejected on the M1 |
| Spatial rearrange | Depth-to-space and space-to-depth in two channel-ordering conventions, plus space-and-batch reshuffles, parameterized by per-axis integer factors | `PixelShuffle`, `PixelUnshuffle`, `ChannelToSpace`, `SpaceToChannel`, `SpaceToBatch`, `BatchToSpace`; `ZinParse<Name>Unit`. Three int32 keys: `FactorX`, `FactorY`, `FactorZ` | All families from the M1 onward; `BatchToSpace` requires batch $N$ divisible by `FactorX` times `FactorY` |
| Range normalization | Maps a tensor to its minimum-to-maximum span per row or per column | `MinMaxNormalization`; `_ANECValidateMinMaxNormLayer`, `_ANECMinMaxNormLayerDescInitialize`. Keys: `Dimension` (Width or Height), `Epsilon` as a float16 bit pattern | Width and Height run; `Dimension` of Channel is arch-gated and rejected on the M1 |
| Local response normalization | Cross-channel response normalization over a channel window | `LocalResponseNormalization`; `_ANECValidateLRNLayer`. `Alpha` is a float16 bit pattern divided internally by `KernelChannel`; only the first `KernelChannel` channels are normalized | All families from the M1 onward |
| Scaled elementwise | A binary elementwise op fused with a scalar scale, $y = s\,(x \mathbin{\mathrm{op}} z)$ | `ScaledElementWise`. Keys: `Type` (Add, Mult, Sub, and the elementwise vocabulary), `Scale` as a float16 bit pattern | All families from the M1 onward |
| Template cross-correlation | Valid cross-correlation of a single-channel map with an unflipped template, $y[i,j] = \sum_{u,v} x[i+u, j+v]\,t[u,v]$ | `CrossCorrelation`. The MIL frontend rejects the op; the netplist `Type` reaches it directly | All families from the M1 onward |
| Three-vector cross product | The cross product of two length-3 vectors held in the channel axis, $\mathrm{cross}(x, z)$ | `CrossProduct`. Inputs shaped D1 C3 H1 W1; the MIL frontend rejects the op | All families from the M1 onward |
| Furthest-point sampling | Greedy L2 furthest-point sampling of up to 1024 centroids from up to 8192 points, seeded at the first point, centroids returned channel-major | `FurthestPointSampling`. Keys: `CentroidCount`, `DistanceMetric` (L2 only on this architecture) | All families from the M1 onward |
| Radius neighborhood search | An L2 ball query returning a points-by-centroids membership matrix, one membership flag per pair | `RadiusSearch`. Two inputs (centroids, points), both D1 C3 H1; key `Radius` | All families from the M1 onward |
| Stereo cost volume | The L1 matching cost per disparity, $\mathrm{cost}[d,x] = \lvert \mathrm{aux}[x] - \mathrm{ref}[x+d] \rvert$, over $R+1$ disparity planes | `CostVolume`. Keys: `DisparityDirection`, `DisparityRange`; requires reference width $W_r \ge W_a + R$ | All families from the M1 onward |
| Re-strided input view | A contiguous offset window $x[\text{Offset} : \text{Offset} + \text{Size}]$ along one named axis, no data movement | `InputView`. Keys: `Dimension`, `Offset`, `Size`, `Step`; gates `InvalidInputView{Dimension,Offset,Size,Step}` | All families from the M1 onward |
| Runtime-offset dynamic slice | A window $x[\text{start} : \text{start} + \text{SliceSize}]$ whose start is bound as a constant or runtime index | `DynamicSlice`; reached by the MIL `slice_by_index` path. Keys: `DynamicSliceAxisOrder`, `DynamicSliceInfo`, `CoordinateInfo`, `PaddingInfo`, `BackgroundValue` | Validator callable on the M1, but the code generator rejects `DynamicSlice` there; runs on later families |
| Tile, concatenate, and reshape utilities | Flatten as an NCHW identity reshape, inference-time dropout as identity, and broadcast of a length-1 axis | `Flatten`, `Dropout` (rate 0), `Broadcast` (keys `Dimension`, `Size`) | All families from the M1 onward |

Table: The hidden-layer catalog. {#tbl:apb-catalog}

## Arch-gated negatives

Three rows above name layers a later chip accepts and the M1 rejects, by the family gates of chapter 12.
`GlobalArgMinMax` is gated to the A15 generation and rejected on the M1.
`MinMaxNormalization` with a Channel reduction is arch-gated and rejected on the M1, while its Width and Height reductions run.
The texture-engine samplers (resize, crop-and-resize, grid resample, and the affine spatial transform) are accepted from the A14 generation and rejected on the M1, where the compiler reports that the affine transform is not supported on this architecture.
They are part of the same gated family but are not authored as netplist Units here.

A second class of rejection is not a family gate but the attested-is-not-reachable rule of chapter 4: the `Sort` and `DynamicSlice` validators are callable on the M1, yet the code generator rejects both, and `TopK` is accepted only outside the $\{3, 4\}$ band.
An authored layer is confirmed by a compile-and-run on the target, not by the presence of its descriptor.

## Validator gate set

Every authored layer passes through one per-layer validator, the `_ANECValidate<Op>Layer` family, of which 55 symbols are exported and 50 are per-layer.
The compiler runs the same validators in two roles: the segmenter dry-runs them through `_ANECValidateNetworkCreate` to decide engine eligibility, and the back-end legalizer re-runs them during a real compile, so the dry-run prediction never drifts from the compile result.
The five non-layer exports are `_ANECValidate`, `_ANECValidateNetworkCreate`, `_ANECValidateMPSModule`, `_ANECValidateMPSModuleCreate`, and `_ANECValidateMutableProcedureInfo`.

Each validator reads a fixed bottom (input-tensor) count and a per-chip feature byte from the hardware-abstraction layer, and rejects with a measured literal string.
[Table](#tbl:apb-validator-gates) reproduces those gates for the validators that guard the authored and bridge-reachable layers, with the bottom count, constraint, and reject string for each.

| Validator | Bottoms | Gate and constraint | Reject string |
| --- | --- | --- | --- |
| `SDPA` | 4 or 5 | key and value same shape; mask broadcast-compatible; scale constant | `SDPA layer must have only 4 or 5(optional mask) inputs` |
| `Conv` | 1 plus weight | kernel within the per-chip range; large kernel W and H multiple of 8; channels divisible by groups | `Invalid conv kernel %s = %zd, It should be in [%zd, %zd]` |
| `MatrixMult` | 2 | depth 1 on both operands; out-C equals A-C; fits the kernel-memory budget | `depth > 1 is not supported for MatMult` |
| `Linear` | 1 plus weight | input rank below 5 | `Linear layer must have only one single input.` |
| `Pool` | 1 | window below input; pad below kernel; mode gated per chip | `Pooling mode "%s" is not available on this ANE architecture.` |
| `Neuron` | 1 | non-linear mode 1 to 46; type in the per-chip list; ReLU-N positive parameters when the gate byte is 0 | `This platform doesn't support Neuron %s` |
| `Reduction` | 1 | reduce-then-square needs feature byte `0x494` (0 through the M1); each axis at most 4 | `square operation after reduction is not supported` |
| `Softmax` | 1 | feature byte `0x815` (0 on older); output Float | `Softmax is not supported by this ANE architecture` |
| `LayerNorm` | 1 | channels divisible by num-groups; grouped form requires depth 1 | `... does not yet support depth > 1` |
| `InstanceNorm` | 1 | feature byte `0x816`; spatial axes only | `InstanceNorm layer not supported for this ANE architecture` |
| `MinMaxNorm` | 1 | feature byte `0x818`; spatial only, the Channel axis arch-gated | (encoded assert, byte `0x818`) |
| `LRN` | 1 | feature byte `0x81a`; channel count of 16 or above fails code generation on the M1 | `LRN is not supported on this architecture.` |
| `ArgMinMax` | 1 | channel-reduce C at most 2048 fp16; pad below kernel; equal left and right, zero front and back | `ArgMinMax layer must have one input` |
| `GlobalArgMinMax` | 1 | feature byte `0x4f2` (1 from the M1 on the bridge route); mode 1 or 2; reduce dimension not 5 | (encoded assert, byte `0x4f2`) |
| `Transpose` | 1 | permutation valid; extent capped 16384 through the M3, 65536 on the M5; last four dimensions only | `NE Input Transpose is not supported for this arch` |
| `Concat` | variadic | match input zero on every non-concat axis; constant positive axis; same layout | `Concat layer must have at least 2 inputs` |
| `Pad` | 1 | H or W axes only; symmetric and reflect modes need texture byte `0x81d` (0 on the M1) | `Channel padding is not supported on ANE` |
| `Broadcast` | 1 | broadcast only from a length-1 axis; depth-axis broadcast needs byte `0x812` | `Broadcast along depth axis is not supported on this architecture` |
| `Gather` | 2 | M1 software envelope: data batch 1, depth 1, index channel 3; texture path from the A14 | `Cannot decompose layer on this architecture` |
| `PixelShuffle` / `PixelUnshuffle` | 1 | depth factor 1; W and H factors in 1, 2, 3, 4, 8; channel divisible by the factor product | `returned invalid:` |
| `SpaceToBatch` / `BatchToSpace` | 1 | factors fully factor into 2, 3, 4, 8; batch divisible by the factor product | `Input batch n = %zd is not divisible by factor x = %d * factor y = %d` |
| `ChannelToSpace` | 1 | the z dimension is not reorganizable | `ChannelToSpace in z dimension is not supported, current factor.z = %d.` |
| `Resize` | 1 | dimension or ratio, not both; sampling axes H and W; texture byte `0x81d` (0 on the M1 takes a software route) | `failed to map resize layer on this arch` |
| `CropResize` | 2 | index format fp16; texture engine from the A14; same coordinate, method, and padding across axes | `Codegen Error: Invalid Texture CropCfg` |
| `AffineTransform` | 2 | matrix fp16; texture byte `0x81d` (0 on the M1) | `affine transform is not supported on this architecture` |
| `Resample` | 2 | warp depth 1; warp channel 1 or 2; texture engine from the A14 | `Channel size in coordinates should be 1 or 2` |
| `Sort` | 1 | direction valid; output fp16 or uint16; validator passes, code generation rejects on the M1 | (passes validation, code-generation reject) |
| `TopK` | 1 | k in the sort-dimension range; the M1 rejects at code generation, and k in 3 or 4 is forbidden on the M5 | (passes validation, code-generation reject) |
| `Dropout` | 1 | feature byte `0x4a9` (1 only from the A15); rate in the half-open unit interval | `Dropout layer is not supported on this architecture.` |
| `Random` | none | feature byte `0x4a9` (1 only from the A15); low below high; output Int8, UInt8, or Float16 | `Random layer is not supported on this architecture.` |
| `RingBufferWriter` | 2 | the writer must connect to a live-state buffer; circular mode arch-gated | `Circular buffer is not supported on this architecture` |
| `NMS` | 2 or more | boxes channel 4; runs only on a CPU or GPU backend, never engine-native | (passes validation, not engine-native) |

Table: The per-layer validator gates. {#tbl:apb-validator-gates}

The validators are public exported symbols, so they are callable from user space, and this is the basis of the precompile predictor: a callable validator marks a schema-gated layer reachable by direct authoring.
A callable validator that accepts the schema does not guarantee the layer compiles, which is why the `Sort`, `TopK`, `DynamicSlice`, and the M1 `CropResize` rows pass the validator and fail at hardware-executable lowering.
One opcode-surface gap holds the other way: the RCAS and Reverse operations have an internal semantics validator but no exported per-layer symbol, so they are not reachable by direct authoring through this route.
