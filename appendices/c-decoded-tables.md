# Appendix C. Decoded reference tables

> This appendix collects the decoded values cited throughout the guide into one reference: enum tokens, struct layouts, status codes, the register-init table, command table, register map, and program-container schema.
> Each section begins with its table and names the research-corpus file that contains the full set when only a representative excerpt is reproduced here.

Each section reproduces the representative and structurally important rows of a larger table.
Where a table runs to hundreds or thousands of rows, the full set is in the research corpus and the section names the file that contains it.
Every value here is read out of an M1/H13 binary by static analysis.

## C.1 Operation-attribute enum tokens to integer values {.unnumbered}

These are the integer codes the compiler resolves attribute string tokens to.
[Table](#tbl:apc-activation-tokens) is the front-end activation token map (`MILOpConverter::NeuronTypeFromString`, 33 entries, miss resolves to 0).

| token | int | token | int |
| --- | --- | --- | --- |
| `relu` | 1 | `gelu` | 23 |
| `leaky_relu` | 2 | `degamma` | 25 |
| `clamped_relu` | 3 | `trunc` | 26 |
| `relu_n` (relu6) | 4 | `round_nearest` | 27 |
| `sigmoid` | 5 | `floor` | 28 |
| `sigmoid_high_precision` | 6 | `ceil` | 29 |
| `tanh` | 7 | `erf` | 31 |
| `silu` (alias `swish`) | 9 | `threshold_relu` | 32 |
| `swish_hard` | 10 | `gamma` | 33 |
| `sqr` | 11 | `inv` | 14 |
| `sqrt` | 12 | `log2` | 15 |
| `rsqrt` | 13 | `exp2` | 16 |
| `elu` | 18 | `exp` | 17 |
| `sin` | 20 | `sign` | 19 |
| `cos` | 21 | `sigmoid_hard` | 22 |

Table: The front-end activation token names and the integer neuron codes they resolve to. {#tbl:apc-activation-tokens}

The serialization dtype enum (`ANECIRDataType`, the `dtype` / `storage_type` attribute) is an 11-code space, given as [table](#tbl:apc-dtype-enum).

| int | type | int | type |
| --- | --- | --- | --- |
| 0 | int4 | 6 | uint16 |
| 1 | uint8 | 7 | int32 |
| 2 | int8 | 8 | uint32 |
| 3 | fp16 | 9 | int64 |
| 4 | fp32 | 10 | uint64 |
| 5 | int16 | | |

Table: The serialization data-type codes and the element types they name. {#tbl:apc-dtype-enum}

The MLIR `symbolize*` enums are a length-dispatched chain of packed integer string compares, fully static, each miss resolving to 0, collected in [table](#tbl:apc-symbolize).

| attribute | token | int |
| --- | --- | --- |
| `padding_style` | `EXPLICIT` / `TF_VALID` / `TF_SAME` / `EXPLICIT_OFFSET` / `ONNX_SAME_LOWER` | 0 / 1 / 2 / 3 / 4 |
| `nearest_rounding_mode` | `round_prefer_ceil` / `round_prefer_floor` / `ceil` / `floor` / `round_to_even` / `round_to_odd` | 0 / 1 / 2 / 3 / 4 / 5 |
| `reduce_op` | `min` / `max` / `sum` / `prod` / `argMin` / `argMax` | 0 / 1 / 2 / 3 / 4 / 5 |
| `data_layout` | `NCHW` / `NHWC` / `OIHW` / `HWIO` / `CHW` / `HWC` / `HW` | 0 / 1 / 2 / 3 / 4 / 5 / 6 |
| `data_layout` | `NCDHW` / `NDHWC` / `OIDHW` / `DHWIO` | 7 / 8 / 9 / 10 |
| `scatter` mode | `add` / `subtract` / `multiply` / `divide` / `min` / `max` / `set` | 0 / 1 / 2 / 3 / 4 / 5 / 6 |
| pool `indices_mode` | `GlobalFlatten1D`..`4D` / `LocalFlatten1D`..`4D` | 0..3 / 4..7 |
| RNN gate activation | `none` / `relu` / `tanh` / `sigmoid` / `hard_sigmoid` / `scaled_tanh` | 0 / 1 / 2 / 3 / 4 / 5 |
| stencil `padding_mode` | `constant` / `mirror` / `mirrorWithEdge` / `clampToEdge` / `zero` / `periodic` / `antiPeriodic` | 0 / 1 / 2 / 3 / 4 / 5 / 6 |
| `pixel_format` | `R8Unorm` / `RG8Unorm` / `RGBA8Unorm` / `BGRA8Unorm` / `R16Float` | 0 / 1 / 2 / 3 / 4 |
| `pixel_format` | `RG16Float` / `RGBA16Float` / `R32Float` / `RG32Float` / `RGBA32Float` | 5 / 6 / 7 / 8 / 9 |
| `arith::RoundingMode` | `to_nearest_even` / `downward` / `upward` / `toward_zero` / `to_nearest_away` | 0 / 1 / 2 / 3 / 4 |
| collective reduction | `sum` / `max` / `min` / `prod` / `mean` | 1 / 2 / 3 / 4 / 5 |

Table: The symbolized attribute tokens and the integer values each one maps to. {#tbl:apc-symbolize}

A second class of enum is in a packed pointer table in the data segment, the internal `Zin` enums the lowering layer dispatches on, listed in [table](#tbl:apc-zin-enums).

| enum | value to token |
| --- | --- |
| `ZinIrPoolingType` | 1 Avg, 2 Max, 3 ChannelMax, 4 Min, 5 ChannelMin, 6 L1, 7 L2, 8 SpatialAndChannelAvg, 9 SpatialAndChannelMax, 10 SpatialAndChannelMin, 11 SpatialArgMax, 12 ChannelArgMax, 13 SpatialArgMin, 14 ChannelArgMin |
| `ZinIrReductionType` | 0 Sum, 1 Min, 2 Max, 3 Avg, 4 SatSum, 5 SatSub, 6 ArgMin, 7 ArgMax, 8 BitwiseAnd, 9 BitwiseOr, 10 BitwiseXor |
| `ZinIrEWType` | 1 Add, 2 Mult, 3 Square, 4 Sub, 5 Power, 6 Div, 7 Max, 8 Min, 9 Abs, 10 EqualZero, 11 NotEqualZero, 12 LessThanZero, 13 LessThanEqualZero, 14 GreaterThanEqualZero, 15 GreaterThanZero, 16 Equal, 17 NotEqual, 18 LessThan, 19 LessThanEqual, 20 GreaterThanEqual, 21 GreaterThan |
| `ZinIrScaledEWType` | 1 Add, 2 Mult, 3 SumSquare, 4 Max, 5 Min |
| `ZinIrPaddingMode` | 1 Zero, 2 Negative, 3 Replication, 5 Symmetric, 6 Reflective, 7 Background, 8 DontCare |
| `ZinIrSamplingMethod` | 0 Linear, 1 NearestNeighbor |
| `ZinIrSamplingGridMode` | 0 AlignedCorners, 1 UnalignedCorners, 2 OffsetCorners, 3 Default, 4 OffsetDefault, 5 OffsetDefaultWithNominalScale, 6 StrictAlignedCorners |
| `ZinIrCoordinateMode` | 0 NonNormalized, 1 NormalizedSymmetric, 2 NormalizedReflect |
| `ZinArgMode` | 1 SpatialArgMin, 2 ChannelArgMin, 3 SpatialArgMax, 4 ChannelArgMax |
| `ZinIrSortDirection` | 0 Invalid, 1 Ascending, 2 Descending |
| `ZinIrTopKType` | 0 Invalid, 1 Min, 2 Max |
| `ZinIrFlattenType` | 1 NCHW, 2 NHWC |
| `ZinIrDimension` | 0 N, 1 D, 2 C, 3 H, 4 W |

Table: The internal `Zin` enum value-to-token maps the lowering layer dispatches on. {#tbl:apc-zin-enums}

The op-class selector is the `ZinUnitType` table, 79 entries, the engine-unit each layer routes to, given in full as [table](#tbl:apc-zinunittype).

| int | unit | int | unit | int | unit |
| --- | --- | --- | --- | --- | --- |
| 1 | Conv | 28 | LayerNormalization | 55 | RandomGenerator |
| 2 | Pooling | 29 | LocalResponseNormalization | 56 | Alias |
| 3 | Concat | 30 | CostVolume | 57 | CrossProduct |
| 4 | ElementWise | 31 | PixelShuffle | 58 | Quant |
| 5 | ScaledElementWise | 32 | PixelUnshuffle | 59 | DeQuant |
| 6 | Neuron | 33 | FurthestPointSampling | 60 | Linear |
| 7 | NeuronCustom | 34 | SpaceToBatch | 61 | RingBufferWriter |
| 8 | GOC | 35 | BatchToSpace | 62 | RingBufferReader |
| 9 | DynamicGOC | 36 | SpaceToChannel | 63 | BatchNorm |
| 10 | ConstMatrixMatrixMult | 37 | ChannelToSpace | 64 | Phi |
| 11 | Flatten | 38 | RadiusSearch | 65 | Condition |
| 12 | Unflatten | 39 | Gather | 66 | WaitForEvent |
| 13 | CrossCorrelation | 40 | AffineTransform | 67 | SignalEvent |
| 14 | KernelRasterizer | 41 | Resize | 68 | NEConv |
| 15 | ArgMinMax | 42 | ResizeAs | 69 | NEMatMul |
| 16 | GlobalArgMinMax | 43 | Resample | 70 | NEPool |
| 17 | InputView | 44 | Padding | 71 | NEBypass |
| 18 | MatrixMultiplication | 45 | Tile | 72 | PEPool |
| 19 | Broadcast | 46 | CropResize | 73 | PEElementWise |
| 20 | Reduction | 47 | DynamicSlice | 74 | PEGOC |
| 21 | Transpose | 48 | PlaneReader | 75 | AllSlice |
| 22 | Reshape | 49 | PlaneWriter | 76 | AllGather |
| 23 | Shape | 50 | Sort | 77 | SDPA |
| 24 | Softmax | 51 | TopK | 78 | AllReduce |
| 25 | InstanceNormalization | 52 | NMS | 79 | FunctionCall |
| 26 | L2Normalization | 53 | MatrixDecomposition | | |
| 27 | MinMaxNormalization | 54 | Dropout | | |

Table: The complete 79-entry `ZinUnitType` op-class table and the engine unit each value selects. {#tbl:apc-zinunittype}

The activation non-linear-mode space is a parallel table, `NonLinearModeToString`, 48 slots indexed directly by the lower hardware mode value, reproduced as [table](#tbl:apc-nonlinearmode).

| idx | mode | idx | mode | idx | mode |
| --- | --- | --- | --- | --- | --- |
| 0 | `none` | 16 | `rsqrt` | 32 | `sin` |
| 1 | `relu` | 17 | `clamped_relu_rsqrt` | 33 | `cos` |
| 2 | `sigmoid` | 18 | `inv` | 34 | `gelu` |
| 3 | `sigmoid_high_precision` | 19 | `sqr` | 35 | `gelu_sigmoid_approximation` |
| 4 | `relu_sigmoid` | 20 | `log2` | 36 | `degamma` |
| 5 | `sigmoid_hard` | 21 | `exp2` | 37 | `round_nearest` |
| 6 | `tanh` | 22 | `exp` | 38 | `trunc` |
| 7 | `clamped_relu` | 23 | `elu` | 39 | `floor` |
| 8 | `prelu` | 24 | `sign` | 40 | `ceil` |
| 9 | `relun` | 25 | `equal_zero` | 41 | `atan` |
| 10 | `swish` | 26 | `not_equal_zero` | 42 | `atan_part1` |
| 11 | `swish_hard` | 27 | `less_than_zero` | 43 | `atan_part2` |
| 12 | `dirac` | 28 | `less_than_equal_zero` | 44 | `erf` |
| 13 | `int` | 29 | `greater_than_equal_zero` | 45 | `thresholded_relu` |
| 14 | `frac` | 30 | `greater_than_zero` | 46 | `gamma` |
| 15 | `sqrt` | 31 | `custom_lut` | 47 | `abs` |

Table: The complete 48-slot `NonLinearModeToString` table indexed by hardware non-linear-mode value. {#tbl:apc-nonlinearmode}

The micro-op opcode space (`ZinIrOpLayerOpCodeType`, 126 codes, `0x00`..`0x7d`) has the codes the task-descriptor builder dispatches on, given in full as [table](#tbl:apc-microop-opcodes).

| op | dec | string | op | dec | string |
| --- | --- | --- | --- | --- | --- |
| `0x00` | 0 | `CONV` | `0x3f` | 63 | `AFFINE_TRANFORM` |
| `0x01` | 1 | `POOL` | `0x40` | 64 | `PLANE_READER` |
| `0x02` | 2 | `SCALE_BIAS` | `0x41` | 65 | `PLANE_WRITER` |
| `0x03` | 3 | `TERNARY_DYNAMIC_GOC` | `0x42` | 66 | `SORT` |
| `0x04` | 4 | `ACTIVATION` | `0x43` | 67 | `TOP_K` |
| `0x05` | 5 | `EW` | `0x44` | 68 | `RCAS` |
| `0x06` | 6 | `SCALED_EW` | `0x45` | 69 | `INDEX` |
| `0x07` | 7 | `CONCAT` | `0x46` | 70 | `NMS` |
| `0x08` | 8 | `SPLIT` | `0x47` | 71 | `DROPOUT` |
| `0x09` | 9 | `COPY` | `0x48` | 72 | `TYPE_CAST` |
| `0x0a` | 10 | `FLATTEN` | `0x49` | 73 | `STOCHASTIC_ROUND` |
| `0x0b` | 11 | `UNFLATTEN` | `0x4a` | 74 | `RANDOM_GENERATOR` |
| `0x0c` | 12 | `CROSS_CORRELATION` | `0x4b` | 75 | `LINEAR` |
| `0x0d` | 13 | `CROSS_PRODUCT` | `0x4c` | 76 | `RINGBUFFER_WRITER` |
| `0x0e` | 14 | `KERNEL_RASTERIZER` | `0x4d` | 77 | `RINGBUFFER_READER` |
| `0x0f` | 15 | `ARG_MIN_MAX` | `0x4e` | 78 | `CONDITION` |
| `0x10` | 16 | `GLOBAL_ARG_MIN_MAX` | `0x4f` | 79 | `PHI` |
| `0x11` | 17 | `MATRIX_MULT` | `0x50` | 80 | `BASICBLOCK_IN` |
| `0x12` | 18 | `BROADCAST` | `0x51` | 81 | `BASICBLOCK_OUT` |
| `0x13` | 19 | `FLATTEN_COMPOSITE` | `0x52` | 82 | `BATCHNORM` |
| `0x14` | 20 | `UNFLATTEN_COMPOSITE` | `0x53` | 83 | `WAIT_FOR_EVENT` |
| `0x15` | 21 | `FPS_WITH_RADIUS_COMPOSITE` | `0x54` | 84 | `SIGNAL_EVENT` |
| `0x16` | 22 | `PIXEL_SHUFFLE_COMPOSITE` | `0x55` | 85 | `ALL_SLICE` |
| `0x17` | 23 | `PIXEL_UNSHUFFLE_COMPOSITE` | `0x56` | 86 | `ALL_GATHER` |
| `0x18` | 24 | `CONV_COMPOSITE` | `0x57` | 87 | `SCALED_DOT_PRODUCT_ATTENTION` |
| `0x19` | 25 | `MATDECOMP_MATMULT_COMPOSITE` | `0x58` | 88 | `ALL_REDUCE` |
| `0x1a` | 26 | `CHANNEL_TO_SPACE_LARGE_FACTOR_COMPOSITE` | `0x59` | 89 | `PEFUSED_ELEMENTWISE` |
| `0x1b` | 27 | `LIVE_IN` | `0x5a` | 90 | `PEFUSED_SECUREFLUSH` |
| `0x1c` | 28 | `LIVEIN_PARAM` | `0x5b` | 91 | `PEFUSED_POOL` |
| `0x1d` | 29 | `CONST_IN` | `0x5c` | 92 | `PEFUSED_GOC` |
| `0x1e` | 30 | `LIVE_STATE` | `0x5d` | 93 | `NEFUSED_CONV` |
| `0x1f` | 31 | `LIVE_OUT` | `0x5e` | 94 | `NEFUSED_KERNEL_RASTERIZER` |
| `0x20` | 32 | `REDUCTION` | `0x5f` | 95 | `NEFUSED_CROSS_CORRELATION` |
| `0x21` | 33 | `ALIAS` | `0x60` | 96 | `NEFUSED_MATMUL` |
| `0x22` | 34 | `REINTERPRET_INNERMOST_DIMENSION` | `0x61` | 97 | `NEFUSED_POOL` |
| `0x23` | 35 | `REINTERPRET_CAST` | `0x62` | 98 | `NEFUSED_EW` |
| `0x24` | 36 | `RESHAPE` | `0x63` | 99 | `NEFUSED_DUAL_SOURCE_EW` |
| `0x25` | 37 | `VIEW` | `0x64` | 100 | `NEFUSED_BYPASS` |
| `0x26` | 38 | `TRANSPOSE` | `0x65` | 101 | `NEFUSED_RCAS` |
| `0x27` | 39 | `SPACE_TO_BATCH` | `0x66` | 102 | `TRANSPOSE_ENGINE_OP` |
| `0x28` | 40 | `BATCH_TO_SPACE` | `0x67` | 103 | `TE_RESAMPLE` |
| `0x29` | 41 | `SPACE_TO_CHANNEL` | `0x68` | 104 | `TE_AFFINE_TRANSFORM` |
| `0x2a` | 42 | `CHANNEL_TO_SPACE` | `0x69` | 105 | `TE_PAD` |
| `0x2b` | 43 | `SOFTMAX` | `0x6a` | 106 | `TE_CROP_RESIZE` |
| `0x2c` | 44 | `INSTANCE_NORM` | `0x6b` | 107 | `TE_SLICE` |
| `0x2d` | 45 | `L2_NORM` | `0x6c` | 108 | `TE_GATHER` |
| `0x2e` | 46 | `MINMAX_NORM` | `0x6d` | 109 | `TE_RESIZE` |
| `0x2f` | 47 | `LAYER_NORM` | `0x6e` | 110 | `TM_WAIT_FOR_EVENT` |
| `0x30` | 48 | `LRN` | `0x6f` | 111 | `TM_SIGNAL_EVENT` |
| `0x31` | 49 | `COST_VOLUME` | `0x70` | 112 | `TM_BRANCH` |
| `0x32` | 50 | `PIXEL_SHUFFLE` | `0x71` | 113 | `TM_FETCH` |
| `0x33` | 51 | `PIXEL_UNSHUFFLE` | `0x72` | 114 | `TM_STORE` |
| `0x34` | 52 | `MATRIX_DECOMPOSITION` | `0x73` | 115 | `TM_OPERATE` |
| `0x35` | 53 | `FPS` | `0x74` | 116 | `TM_USER_SLOT_LOAD` |
| `0x36` | 54 | `RS` | `0x75` | 117 | `DMA_CONVERT` |
| `0x37` | 55 | `RESAMPLE` | `0x76` | 118 | `QUANT` |
| `0x38` | 56 | `GATHER` | `0x77` | 119 | `DEQUANT` |
| `0x39` | 57 | `TILE` | `0x78` | 120 | `SNE_COND` |
| `0x3a` | 58 | `SLICE` | `0x79` | 121 | `SNE_GOC` |
| `0x3b` | 59 | `PAD` | `0x7a` | 122 | `CCDMA_CONST` |
| `0x3c` | 60 | `RESIZE` | `0x7b` | 123 | `CCDMA_MEMORY` |
| `0x3d` | 61 | `RESIZEAS` | `0x7c` | 124 | `SPILL_FILL_DUMMY` |
| `0x3e` | 62 | `CROP_RESIZE` | `0x7d` | 125 | `INVALID` |

Table: The complete 126-entry micro-operation opcode table and the layer-kind name each value dispatches on. {#tbl:apc-microop-opcodes}

The opcode `0x3f` is spelled `AFFINE_TRANFORM` in the binary, a vendor source typo preserved on the wire.

## C.2 Operation-attribute schema and IOKit external-method struct layouts {.unnumbered}

The attribute schema is string-keyed: the token is the wire encoding the compiler matches on, and most integer constants are not recoverable statically.
The converter recognizes 171 literal attribute keys, of which roughly 140 are op-facing.
[Table](#tbl:apc-attribute-keys) gives representative keys with their value types, meanings, and wire encodings.

| key | type | meaning | value encoding |
| --- | --- | --- | --- |
| `activation` | enum | neuron mode | token into the 22-field PWL descriptor |
| `strides` | int[] | per-axis stride | NDCHW int array; deconv restricted to {1,2} |
| `groups` | int | group count | channel-wise requires `groups == out.C` |
| `padding_mode` | enum | fill rule | `constant` / `reflect` / `replicate` / `symmetric` |
| `weights_layout` | enum | weight axis order | `NCHW` / `NHWC` / `OIHW` / `HWIO`; weight buffer is `MACI` |
| `compressed` | bool/enum | weight compression | format set by the MIL-op-count contract |
| `interleave` | int | channel-tiling quantum | one of {1,2,3,4,8} |
| `epsilon` | float | norm stability | scalar |

Table: Representative operation-attribute keys with their value types, meanings, and wire encodings. {#tbl:apc-attribute-keys}

The host-to-kernel IOKit dispatch key is the (selector, struct-size) tuple, not the selector alone.
[Table](#tbl:apc-iokit-selectors) gives the control-client selectors with their method names and decoded input and output struct sizes.

| sel | method | in-struct | out/scalar |
| --- | --- | --- | --- |
| 0 | `ANE_DeviceOpen` | 104 | 104 |
| 2 | `ANE_ProgramSendRequest` | 2376 + 1 scalar | 40 async |
| 3 | `ANE_ProgramCreate` | 32 | 0 |
| 4 | `ANE_ProgramPrepare` | 56 | 56 |
| 6 | `ANE_ProgramDestroy` | 16 | 0 |
| 7 | `ANE_GetStatus` | 0 | 32 |
| 8 | `ANE_ProgramCreateInstance` | 32 | 0 |
| 10 | `ANE_GetVersion` | 0 | 1 scalar |
| 21 | `ANE_ProgramInputsReady` | 3104 | 0 |
| 22 | `ANE_MemoryMapRequest` | 2080 + 1 scalar | 1 scalar |

Table: The IOKit control-client selectors with their method names and decoded input and output struct sizes. {#tbl:apc-iokit-selectors}

The `ANEDeviceOpen` shared in/out buffer (104 bytes, selector 0) decodes by byte offset as [table](#tbl:apc-anedeviceopen).

| offset | field |
| --- | --- |
| `+0x00` | usage type (1 standard, 2 unsupported) plus session token |
| `+0x08` | callback function pointer |
| `+0x10` | receiver context pointer |
| `+0x18` | timeout `0x2710` (10000) |
| `+0x48` | version pair `32, 256` |
| `+0x50` | NumANEs `0, 1` |

Table: The field layout of the ANEDeviceOpen shared input and output buffer by byte offset. {#tbl:apc-anedeviceopen}

The full 171-key attribute corpus, the decoded enum-value tables, and the per-selector field layouts for the HW direct-path client are in the research corpus.

## C.3 Numeric error, status, and return codes {.unnumbered}

The ANE stack has no flat numeric status enum.
The fixed numeric values that exist are the IOKit return constants and the firmware magic and sentinel words, the first of which [table](#tbl:apc-ioreturn) gives with their meanings on the dispatch path.

| macro | hex | ANE path meaning |
| --- | --- | --- |
| `kIOReturnSuccess` | `0x00000000` | success |
| `kIOReturnError` | `0xe0000001` | general failure |
| `kIOReturnBusy` | `0xe0000007` | gate or command busy |
| `kIOReturnNoMemory` | `0xe00002bd` | allocation failure |
| `kIOReturnNoResources` | `0xe00002be` | out of resources, queue or slot exhaustion |
| `kIOReturnNotPrivileged` | `0xe00002c1` | privilege check failed |
| `kIOReturnBadArgument` | `0xe00002c2` | typed-args validation failure |
| `kIOReturnUnsupported` | `0xe00002c7` | disabled or stub path; also what a gated feature returns when its entitlement is absent |
| `kIOReturnNotReady` | `0xe00002d0` | device or channel not ready |
| `kIOReturnAborted` | `0xe00002eb` | request aborted |
| `kIOReturnNotFound` | `0xe00002f0` | program or process handle not found |
| `kIOReturnTimeout` | `0xe0000404` | firmware op timed out |

Table: The IOKit return constants and their meanings on the engine dispatch path. {#tbl:apc-ioreturn}

At every layer the error surface is name-based or message-based.
The client-visible surface above IOKit is an error factory that wraps the lower-layer code into a structured error across four domains, named `errorDomainCompiler`, `errorDomainEspresso`, `errorDomainGeneric`, and `errorDomainVirtIO`.
Its factory methods are the taxonomy: a generic wrapper, a missing-code-signing form, program-load and new-instance-load forms that hold the lower-layer code, surface map and unmap forms, and a virtualization-kernel form.
The single most look-up-worthy client-visible value is `0xe00002c7` (`kIOReturnUnsupported`), returned on a disabled or unsupported path and when a gated feature's entitlement is absent.

[Table](#tbl:apc-firmware-magic) gives the fixed firmware magic words and sentinel constants with their meanings.

| constant | hex | meaning |
| --- | --- | --- |
| package magic | `0x414E4548` (`ANEH`) | loader package header |
| program magic | `0x414E4550` (`ANEP`) | loader program header |
| section magic | `0x414E4553` (`ANES`) | loader section header |
| AFPP control magic | `0x55AA55AA` | AFPP control struct |
| checksum-valid sentinel | `0xFFFFFFFF` | command checksum initialized and valid |
| invalid id | `0xFFFFFFFF` | `ECSneCmdId_Invalid`, unbound program or process |
| padding | `0x00000000` | command padding must be zero |
| power-status byte | `0xFF` / `0x00` | fully on / fully off |

Table: The fixed firmware magic words and sentinel constants with their meanings. {#tbl:apc-firmware-magic}

This section shows the three loader magic words as 32-bit integers; on disk the bytes are little-endian, so a raw byte scan finds `HENA`, `PENA`, and `SENA` (the characters of `ANEH`, `ANEP`, `ANES` reversed).
The firmware-to-host notification names, inline `status=0x%x` print sites, AArch64 and L2C fault-register dump fields, and compiler diagnostic categories are in the research corpus.

## C.4 The tunable register-init table {.unnumbered}

The per-chip register init is a sequence of 12-byte `(offset, mask, value)` records, each applied as a masked read-modify-write, `reg = (reg & ~mask) | value`, where `reg` is the block MMIO base plus the offset.
The M1 (ASC AscChinook) firmware has 1994 records across 10 named MMIO blocks, each block reached through a 32-byte descriptor of name, MMIO base, record pointer, and count, the blocks and their counts given in [table](#tbl:apc-reginit-blocks).

| block name | MMIO base | records |
| --- | --- | --- |
| `ASC_CHINOOK` | `0x2_6b00_0000` | 24 |
| `ASCWRAP` | `0x2_6b40_0000` | 2 |
| `sneCtrl` | `0x2_6b84_0000` | 15 |
| `ANE` | `0x2_6bc0_0000` | 47 |
| `aneDpePpt` | `0x2_6b8e_c000` | 304 |
| `aneDpePptAccp0` | `0x2_6b8e_d000` | 528 |
| `aneDpePptAccp1` | `0x2_6b8e_e000` | 528 |
| `aneDpePptAccp2` | `0x2_6b8e_f000` | 528 |
| `aneDpeSys` | `0x2_6b8f_0000` | 9 |
| `aneDpePpt_soc_dpe_lee` | `0x2_6b8f_4000` | 9 |

Table: The named MMIO register-init blocks with their bases and record counts. {#tbl:apc-reginit-blocks}

[Table](#tbl:apc-reginit-records) gives representative records, one or more per block, with their address, mask, value, and meaning.

| regAddr | mask | value | meaning |
| --- | --- | --- | --- |
| `0x2_6b14_0020` | `0xf80000` | `0x780000` | ASC clock or PLL divider field set to 15 |
| `0x2_6b40_080c` | `0x6000_0001` | `0x6000_0001` | fabric clock and QoS enable |
| `0x2_6b84_0028` .. `0044` | `0x8fff_c000` | `0x8fff_c000` | 8 identical SNE QoS and credit words, one per set |
| `0x2_6bc0_d014` .. `fec4` | `0x1` | `0x1` | 32 per-tile MAC clock and power enables |
| `0x2_6bc1_400c` | `0xffff_ff00` | `0x4010_1000` | DMA descriptor base and config word |
| `0x2_6b8e_c000` | `0xffff` | `0x267e` | peak-power-tracking base budget word |
| `0x2_6b8e_c42c` | `0x3fff` | `0x0` | DPE trailing-control reg, armed live to `0x3fff` |
| `0x2_6b8e_d000` | `0xffff_ffff` | `0x0077_3594` | per-counter energy scale coefficient (7811476) |
| `0x2_6b8f_0014` | `0xffff_ffff` | `0x0000_23e1` | DPE config and period word |
| `0x2_6b8f_0038` | `0xffff_ffff` | `0x0003_2dcc` | DPE accumulation window and divisor (207820) |
| `0x2_6b8f_4000` | `0x1e` | `0xc` | SoC-level LEE control field |

Table: Representative register-init records with their address, mask, value, and meaning. {#tbl:apc-reginit-records}

The 32 `mask=1 value=1` records at a regular stride are direct evidence of the 32-tile MAC array geometry, each tile individually clock and power gateable.
The DPE system block has seven ascending sampling thresholds (25, 50, 70, 85, 95, 105, 115), and the SoC leakage-estimation block has eight (10, 22, 39, 64, 89, 121, 164, 189): the firmware-side breakpoints of the power model.
All 1994 decoded records are in the research corpus.

## C.5 The `CSNE_CMD_*` numeric command table {.unnumbered}

The host-to-firmware command set is 93 entries, numbered `0x00` through `0x5c`, indexed by `eCSneCmdId` into the firmware command-name string table, whose index is the numeric command identifier.
`0xFFFFFFFF` is the no-command sentinel.
[Table](#tbl:apc-command-table) gives the full set; its `dir` column is `H->FW` for a host request and `FW->H` for a firmware notification, and the subsystem codes are lifecycle, power, secure, program, execution, cache, ipc, buffer, property, and stats.

| id | name | dir | subsystem | purpose |
| --- | --- | --- | --- | --- |
| `0x00` | `STOP` | `H->FW` | lifecycle | stop the controller |
| `0x01` | `RESET` | `H->FW` | lifecycle | reset controller state |
| `0x02` | `CONFIG_GET` | `H->FW` | property | read config blob |
| `0x03` | `PRINT_ENABLE` | `H->FW` | stats | enable firmware print |
| `0x04` | `REG_FILE_LOAD` | `H->FW` | lifecycle | load a register file |
| `0x05` | `BUILDINFO` | `H->FW` | lifecycle | return firmware build string |
| `0x06` | `TIMEPROFILE_START` | `H->FW` | stats | begin time-profiling |
| `0x07` | `TIMEPROFILE_STOP` | `H->FW` | stats | stop time-profiling |
| `0x08` | `TIMEPROFILE_SHOW` | `H->FW` | stats | dump profile |
| `0x09` | `FW_RUN_MODE` | `H->FW` | lifecycle | select firmware run mode |
| `0x0a` | `POWER_DOWN` | `H->FW` | power | full power-down |
| `0x0b` | `SET_SNE_PMU_BASE` | `H->FW` | power | set PMU MMIO base |
| `0x0c` | `SET_SNE_RPC_CHECK_CMD` | `H->FW` | property | RPC sanity-check command |
| `0x0d` | `RPC_ENABLE` | `H->FW` | property | enable the back-channel RPC channel |
| `0x0e` | `PLATFORM_INFO` | `H->FW` | lifecycle | platform descriptor |
| `0x0f` | `BOOT` | `H->FW` | lifecycle | bring firmware to booted state |
| `0x10` | `PING` | `H->FW` | lifecycle | liveness probe |
| `0x11` | `CONFIG_GET_EXT` | `H->FW` | property | extended config read |
| `0x12` | `POWER_DEVICE_ON` | `H->FW` | power | power the device on |
| `0x13` | `POWER_DEVICE_OFF` | `H->FW` | power | power device off |
| `0x14` | `IPC_ENDPOINT_SET` | `H->FW` | ipc | bind an IPC endpoint |
| `0x15` | `IPC_ENDPOINT_UNSET` | `H->FW` | ipc | unbind endpoint |
| `0x16` | `CH_INFO_GET` | `H->FW` | buffer | channel info query |
| `0x17` | `CH_BUFFER_RECYCLE_MODE_SET` | `H->FW` | buffer | set buffer-recycle mode |
| `0x18` | `CH_BUFFER_RECYCLE_START` | `H->FW` | buffer | start recycling |
| `0x19` | `CH_BUFFER_RECYCLE_STOP` | `H->FW` | buffer | stop recycling |
| `0x1a` | `CH_BUFFER_RETURN` | `H->FW` | buffer | return one pooled buffer |
| `0x1b` | `CH_BUFFER_POOL_CONFIG_GET` | `H->FW` | buffer | read buffer-pool config |
| `0x1c` | `CH_BUFFER_POOL_CONFIG_SET` | `H->FW` | buffer | configure buffer-pool |
| `0x1d` | `CH_DATA_FILE_LOAD` | `H->FW` | buffer | stream a data file over channel |
| `0x1e` | `CH_PROPERTY_WRITE` | `H->FW` | property | write a register or property |
| `0x1f` | `CH_PROPERTY_READ` | `H->FW` | property | read a register or property |
| `0x20` | `TRACE_ENABLE` | `H->FW` | stats | enable tracing |
| `0x21` | `RESOURCE_INFO_GET` | `H->FW` | lifecycle | query engine resources |
| `0x22` | `STATS_BUFFER_SIZE_GET` | `H->FW` | stats | compute required stats-buffer size |
| `0x23` | `SUSPEND` | `H->FW` | lifecycle | suspend engine |
| `0x24` | `DSID_SET` | `H->FW` | cache | set data-set identifiers for prefetch |
| `0x25` | `MCACHE_SIZE_GET` | `H->FW` | cache | query memory-cache size |
| `0x26` | `SECURE_MODE_START` | `H->FW` | secure | enter secure mode |
| `0x27` | `SECURE_MODE_STOP` | `H->FW` | secure | leave secure mode |
| `0x28` | `SET_SNE_PMU_BASE2` | `H->FW` | power | version 2 PMU base set |
| `0x29` | `IPC_ENDPOINT_SET2` | `H->FW` | ipc | version 2 endpoint bind |
| `0x2a` | `IPC_ENDPOINT_UNSET2` | `H->FW` | ipc | version 2 unbind |
| `0x2b` | `CH_DATA_FILE_LOAD2` | `H->FW` | buffer | version 2 data-file load |
| `0x2c` | `SET_DYNAMIC_POWERGATE` | `H->FW` | power | configure dynamic clock and power gating |
| `0x2d` | `ANE_DEFAULT_SETTING_SET` | `H->FW` | lifecycle | bulk default-settings push |
| `0x2e` | `INIT_SHARED_EVENT_INFO` | `H->FW` | ipc | initialize shared-event table |
| `0x2f` | `EXCLAVE_MODE_START` | `H->FW` | secure | enter exclave mode (stubbed on H13) |
| `0x30` | `EXCLAVE_MODE_STOP` | `H->FW` | secure | leave exclave mode |
| `0x31` | `QUIESCE_STATE` | `H->FW` | lifecycle | drain in-flight work |
| `0x32` | `CPU_LOAD_GET` | `H->FW` | stats | sample CPU load |
| `0x33` | `SECURE_MODE_RESUME_TRANSITION` | `H->FW` | secure | resume a paused secure transition |
| `0x34` | `CH_ERROR_NOTIFICATION` | `FW->H` | stats | error notification |
| `0x35` | `CH_POWER_CONTROL` | `H->FW` | power | channel-level power control |
| `0x36` | `CH_SIGNPOST_NOTIFICATION` | `FW->H` | stats | 32-bit signpost notification |
| `0x37` | `CH_SIGNPOST_NOTIFICATION_GROUP` | `FW->H` | stats | grouped 32-bit signpost |
| `0x38` | `CH_RESET_NOTIFICATION` | `FW->H` | stats | reset notification |
| `0x39` | `CH_SIGNPOST64_NOTIFICATION` | `FW->H` | stats | 64-bit signpost |
| `0x3a` | `CH_SIGNPOST64_NOTIFICATION_GROUP` | `FW->H` | stats | grouped 64-bit signpost |
| `0x3b` | `CPU_LOAD_NOTIFICATION` | `FW->H` | stats | CPU-load notification |
| `0x3c` | `TM_SYNC_ERR_NOTIFICATION` | `FW->H` | stats | tile-manager sync error |
| `0x3d` | `LOAD_PROGRAM` | `H->FW` | program | load a compiled program into a slot |
| `0x3e` | `UNLOAD_PROGRAM` | `H->FW` | program | unload a program |
| `0x3f` | `CREATE_PROCESS` | `H->FW` | program | instantiate a process for a program |
| `0x40` | `TERMINATE_PROCESS` | `H->FW` | program | tear down a process |
| `0x41` | `PROCEDURE_CALL` | `H->FW` | execution | baseline network invocation |
| `0x42` | `LOAD_AFPP` | `H->FW` | program | load AFPP prefetch program |
| `0x43` | `UNLOAD_AFPP` | `H->FW` | program | unload AFPP |
| `0x44` | `PROGRAM_INTERFACE_VERSION_CHECK` | `H->FW` | program | negotiate program-interface version |
| `0x45` | `PROCEDURE_CALL_CACHE_REQUEST` | `H->FW` | cache | install a resident cache request |
| `0x46` | `PROCEDURE_CALL_TRIGGER_CACHE_REQUEST` | `H->FW` | cache | fire an installed cache request |
| `0x47` | `PROCEDURE_CALL_RECYCLE_OUTPUT_BUFFER` | `H->FW` | cache | return a consumed output buffer |
| `0x48` | `PROCEDURE_CALL_INVALIDATE_CACHE_REQUEST` | `H->FW` | cache | destroy a cache request |
| `0x49` | `PROCEDURE_CALL_WITH_CUSTOM_BARS` | `H->FW` | execution | proc-call with custom barrier array |
| `0x4a` | `PREMAP_BUFFER` | `H->FW` | cache | pre-map an inference-property buffer |
| `0x4b` | `PROCEDURE_CALL_CACHE_REQUEST_WITH_CUSTOM_BARS` | `H->FW` | cache | cache request with custom bars |
| `0x4c` | `PROCEDURE_CALL_CACHE_REQUEST_WITH_SHARED_EVENTS` | `H->FW` | cache | cache request with shared events |
| `0x4d` | `FORCE_DISABLE_CACHE_REQUESTS` | `H->FW` | cache | global cache-request disable |
| `0x4e` | `PROCEDURE_CALL_WITH_SIGNAL_EVENTS` | `H->FW` | execution | proc-call with wait and signal events |
| `0x4f` | `SET_ACTIVE_CACHE_REQUEST_IN_GROUP` | `H->FW` | cache | select active member of a cache-request group |
| `0x50` | `PROGRAM_EVENT` | `FW->H` | program | per-program event notification |
| `0x51` | `USER_EVENT` | `FW->H` | stats | user event marker |
| `0x52` | `DBG_EVENT` | `FW->H` | stats | debug event |
| `0x53` | `DATA_CHAINING_EVENT` | `FW->H` | cache | data-chaining stage completion |
| `0x54` | `PREFETCH_DSID_EVENT` | `FW->H` | cache | prefetch completion |
| `0x55` | `SECURE_MODE_EVENT` | `FW->H` | secure | secure-mode state-change event |
| `0x56` | `REQUEST_PROGRAM_ID` | `H->FW` | program | allocate a program-id slot |
| `0x57` | `RETURN_PROGRAM_ID` | `H->FW` | program | free a program-id slot |
| `0x58` | `REQUEST_PROCESS_ID` | `H->FW` | program | allocate a process-id |
| `0x59` | `RETURN_PROCESS_ID` | `H->FW` | program | free a process-id |
| `0x5a` | `INFERENCE_CALL` | `H->FW` | execution | high-level inference submission |
| `0x5b` | `BACK_CHANNEL_RPC` | `FW<->H` | property | firmware-initiated back-channel RPC |
| `0x5c` | `DEBUG_COMMAND_DATA_CHECK` | `H->FW` | stats | validate command-data integrity |

Table: The complete 93-entry host-to-firmware command table with name, direction, subsystem, and purpose. {#tbl:apc-command-table}

Two strings the prior corpus counted have no numeric identifier: `CSNE_CMD_START` is a standalone lifecycle log alias, and `CSNE_CMD_IPC_ENDPOINT_TYPE_DATA_CHAINING` is an endpoint-type enum value rather than a command.
The per-call numeric limits the command bodies enforce on the M1 are fixed.
The dispatch caps are at most 16 signal events per call, at most 32 custom barriers on the wire (128 in the program container), at most 128 custom execute-order entries, at most 16 trigger input buffers, and fewer than 2 active shared events.
Priority levels run 0 through 7, split into a privileged band of 0 and 1 and a normal band of 2 through 7.
The single dispatch path takes exactly one output buffer set, one task-descriptor partition, and one engine request per list.

[Table](#tbl:apc-command-header) gives the fixed-header layout (`sCSneControllerCmdHdr`) that prefixes every ring message, with each field's offset, width, and meaning.

| field | offset | width | meaning |
| --- | --- | --- | --- |
| `id` | `0x00` | `u32` | the `eCSneCmdId` selector |
| `size` | `0x04` | `u32` | byte length of the command body |
| `priority` | `0x08` | `u32` | scheduling band, 0..7 (0..1 realtime, 2..7 normal) |
| `programId` | `0x0c` | `i32` | loaded-program slot, `-1` invalid |
| `processId` | `0x10` | `i32` | per-program process instance, `-1` none |
| `procedureId` | `0x14` | `u32` | index into the program's procedure table |

Table: The fixed command-header fields with their offsets, widths, and meanings. {#tbl:apc-command-header}

The full 93-entry table with file offsets, the decoded request structs, and the per-call numeric limits are in the research corpus.

## C.6 The task-descriptor hardware register map {.unnumbered}

A captured `ane_reg` record is a `(regAddr, regValue)` pair whose low address selects one of 7 aperture groups.
[Table](#tbl:apc-aperture-map) is the aperture map that converts a raw address to a group, with the image base and window of each.

| regAddr range | group | image base | window |
| --- | --- | --- | --- |
| `< 0x4c` | G1 dimensions | `0xf4` | 19 words |
| `0x4100` .. `0x4177` | G3 elementwise / planar / pad | `0x264` | 30 words |
| `0x4500` .. `0x4537` | G4 L2 / texture | `0x2e4` | 14 words |
| `0x4900` .. `0x492b` | G5 kernel-fmt / op-mode | `0x324` | 11 words |
| `0x4d00` .. `0x4e13` | G2 tile DMA | `0x148` | 69 words |
| `0x5100` .. `0x5153` | G6 L2-result | `0x358` | 21 words |
| `0x5500` .. `0x5587` | G0 kernel / common | `0x24` | 34 words |

Table: The register-address ranges and the aperture group, image base, and window each maps to. {#tbl:apc-aperture-map}

[Table](#tbl:apc-g1-fields) gives representative G1 dimension fields, which also pack the format and control bits, with the bit range and width of each.

| regAddr | field | bits | width | meaning |
| --- | --- | --- | --- | --- |
| `0x00` | Win | [14:0] | 15 | input tile width |
| `0x02` | Hin | [14:0] | 15 | input tile height |
| `0x0c` | Cin | [16:0] | 17 | input channels, max 131071 |
| `0x10` | Cout | [16:0] | 17 | output channels |
| `0x10` | CommonInFmt | [1:0] | 2 | source-1 element format |
| `0x10` | CommonOutFmt | [5:4] | 2 | output element format |
| `0x28` | numGroups | [12:0] | 13 | convolution groups |
| `0x38` | CommonTaskType | [7:4] | 4 | hardware task class (9 valid) |

Table: Representative G1 dimension register fields with their bit ranges, widths, and meanings. {#tbl:apc-g1-fields}

To invert a raw value: DMA strides are 26-bit signed at bits [31:6]; L2-result base and strides are 17-bit at bits [20:4]; a full device address is `(hi << 32) | lo` with `lo` 64-byte aligned and `hi` 10 bits, capped at 42 bits.
The complete inventory of roughly 190 register fields across the 7 groups, 11 reloc slots, and on-M1 stubbed engines (CCDMA, atomic scatter, LDTID) is in the research corpus.

Each operation descriptor in the task-descriptor stream has a 32-bit opcode word.
[Table](#tbl:apc-opcode-words) gives the words decoded from a live M1 program for three operations.

| operation | opcode word |
| --- | --- |
| convolution | `0x5042a063` |
| reduce-mean | `0x5000a021` |
| matrix multiply | `0x5000b021` |

Table: The version-7 / H13 codegen opcode words for three operations, with the high half-word shared and the low 16 bits selecting the operation. {#tbl:apc-opcode-words}

## C.7 The `.e5` FlatBuffer schema {.unnumbered}

The program container is a FlatBuffer whose root table holds four fields, the schema given as [listing](#lst:apc-e5-schema).
The schema reconstructs from the serializer method set and the wire bytes, since the binary strips the reflection schema, and round-trips cleanly through the FlatBuffers tool to 23 tables, 4 enums, and 1 union.
The data-type and op-type enums are recovered by name; the numeric ordinals are inferred.

```fbs
namespace E5RT.fb;

enum TensorDataType : int {
  Invalid = 0, Float16 = 1, Float32 = 2, Int8 = 3, UInt8 = 4,
  Int16 = 5, Int32 = 6, Int4 = 7, Bool = 8, E4M3 = 9, E5M2 = 10
}

enum OpType : int {
  Cast = 0, AneInference = 1, EirInference = 2, CpuInference = 3,
  BnnsCpuInference = 4, MlcCpuInference = 5, MpsGraphInference = 6,
  E5MinimalCpu = 7, Quant = 8, Dequant = 9, Barrier = 10, JitCall = 11
}

table TensorDescriptor {
  dim:[ulong];
  stride:[ulong];
  width:ulong;
  height:ulong;
  channels:ulong;
  batch_number:ulong;
  sequence_length:ulong;
  stride_width:ulong;
  stride_height:ulong;
  stride_channels:ulong;
  stride_batch_number:ulong;
  stride_sequence_length:ulong;
  storage_type:TensorDataType;
  component_pack:int;
}

table BuildInfoEntry { key:string; value:string; }
table BuildInfo { entries:[BuildInfoEntry]; }   // 7 key-value pairs in the sample

table AliasSymbol { name:string; symbol_index:uint; addr_offset:uint; }

table IOPort { name:string; byte_size:ulong; aperture_va:ulong; }

table Operand { descriptor:TensorDescriptor; }

table CastAttrs { src_dtype:TensorDataType; dst_dtype:TensorDataType; component_pack:int; }

table AneInferenceAttrs {
  procedure_name:string;
  anehash:string;
  program_symbol:string;
  intermediate_buffer_handle:uint;
  compiler_options:string;
}

table Operation {
  name:string;
  op_type:OpType;
  inputs:[uint];
  outputs:[uint];
  arg_frame:string;        // __arg_frame section reference
  attrs_section:string;    // __op_attrs section reference
}

table Block { name:string; operations:[Operation]; }

table Function { name:string; anehash_path:string; blocks:[Block]; }

table Section { name:string; kind:int; }

table E5Program {
  symbol_names:[string];   // field[0] name vector
  build_info:BuildInfo;    // field[1] 7-entry sub-table
  sections:[Section];      // field[2] 6-entry section vector
  format_version:int;      // field[3] inline scalar == 4
}

root_type E5Program;
```

Listing: The reconstructed program-container FlatBuffer schema: root table, type enums, tensor descriptor, section tables, and operation structure. {#lst:apc-e5-schema}

The whole fused graph collapses to a single `AneInference` operation, with the surrounding `Cast` operations holding the input and output dtype conversion.
The validation against the round-9 `H13C.e5` sample reads the four root fields, the seven build-info pairs (`built-for-profiling`, `input-file-path`, the component versions, and `on-device-compilation`), and the operation chain `Cast`, `AneInference`, `Cast` straight out of the bytes with no contradiction.
The enum ordinals, the `e4m3` and `e5m2` dtypes, segment-chaining fields, and field sets of the ten op-attribute tables other than `CastAttrs` and `AneInferenceAttrs` are inferred rather than byte-confirmed in this single-segment sample.
