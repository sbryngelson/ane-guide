# 26. Hidden layers and direct netplist authoring

> The compiler validates and lowers a native catalog of 45 hardware layer descriptors, and the model converter surfaces only the subset its public operation set covers.
> Authoring the network description directly reaches the rest, including fused attention, ranking, spatial rearrangement, and geometry layers, each cutting its own dispatch segment.
> Authoring does not bypass a family gate: the texture-engine samplers and whole-tensor arg-min and arg-max are accepted from the A14 and A15 and rejected on the M1, and a layer is confirmed only by a compile-and-run on the target.

The engine implements more layer kinds than the high-level conversion path emits.
Authoring the network description directly reaches the rest; this chapter gives the method, names the classes it reaches, and marks the ones a later chip accepts and the M1 rejects.

The machinery this rests on is decoded in full in the back half: the `.espresso.net` program format in chapter 23, compiler's layer parsers and validators in chapter 22, and per-family capability gates in chapter 24.
Here it is enough that the format is hand-authorable and that each layer has a validator and a family gate; Appendix B is the full per-layer schema.

## Framework and native catalogs

The compiler has two parallel layer catalogs.
The framework-level set is the engine-agnostic kernel abstraction, about 190 layer types, of which some run only on the host or the GPU.
Below it is the native hardware set: 45 descriptor structs, each with a constructor named `_ANEC<Name>LayerDescInitialize` and a checker named `_ANECValidate<Name>Layer`, and this set is the list of operations the engine silicon runs.

The conversion path from a trained model walks the documented intermediate-language operation set [AppleCoreMLTools] and emits only the native layers that set maps onto.
Several native descriptors have no operation in that set, so the converter never produces them, and a model passed through it decomposes the work into the operations it does know.
Fused attention is the worked case: the converter always splits scaled dot-product attention into a matrix multiply, scale, softmax, and second matrix multiply, because its operation set has no atom that lowers to the single fused descriptor.
The descriptor and its validator are present in the compiler the whole time.
Reaching them means handing the compiler a network description that names the layer directly, rather than one produced by decomposing a model.

## Method

A network expressed for the compiler is a property list, the `.espresso.net` representation the compiler accepts alongside the intermediate language.
Its layers are dictionary entries the compiler calls Units, each holding a `Type` tag, a `Bottom` wiring list, an output type, and a `Params` sub-dictionary of typed attributes, all under a `ProcedureList` of callable entry points that name the `InputList`, `OperationList`, and `OutputList`.
The representation is hand-authorable, which is what makes the hidden layers reachable.

The reusable method has four steps.
Author one Unit whose `Type` is the native layer name and whose `Params` hold the descriptor attributes that layer's parser expects.
Supply the wiring and any constant weight blobs the layer needs.
Compile the description through the runtime, which lowers the Unit to its `ANEC<Name>LayerDesc`, runs the matching validator, and assigns it to an engine.
Then load, bind buffers to the named ports, and dispatch, exactly as chapter 6 gives for any compiled program.
A raw native descriptor enters the network without re-expressing it as a framework kernel through the tunneled-unit path: the path passes the descriptor through the framework layer untouched, with its float16 and integer operands intact, so a `Type` of `SDPA` or `CostVolume` reaches the silicon directly.

The descriptor attributes are read out of the compiler's per-layer parsers, the `ZinParse<Name>Unit` routines, and cross-referenced constant-string tables.
A required key the parser does not find raises a parse error such as `InvalidParamSyntax`, so an authored layer either has the exact attribute set the parser expects or fails at compile time rather than at dispatch.

## Classes reached this way

The reachable native layer kinds fall into several groups.
Fused attention is the scaled dot-product attention descriptor, four or five operands of query, key, value, and constant scale, with an optional additive mask as the fifth that holds the causal and decode cases as data.
Ranking and selection cover the top-k, sort, and argument-minimum-and-maximum descriptors, including the whole-tensor argument form, with their integer index outputs returned float16-encoded and exact for the index ranges these layers produce.
Spatial rearrangement holds the pixel-shuffle and pixel-unshuffle pair, channel-and-space pair, and space-and-batch pair, each parameterized by per-axis integer factors and distinguished by channel-ordering convention rather than being aliases.
Three further normalizations appear: the range normalization that maps a tensor to its minimum-to-maximum span, local response normalization, and per-channel affine gain-offset control in static and runtime-tensor forms.
Geometry and point-cloud work draws on the template cross-correlation, the three-vector cross product, furthest-point sampling, radius neighborhood search, and the stereo cost volume.
Data movement completes the set with the re-strided input view, runtime-offset dynamic slice, and tile and concatenate descriptors.
The full catalog of these layers, with each layer's `Type` tag, its descriptor, and its `Params` schema, is Appendix B.

## A native layer descriptor

The fused-attention descriptor is the clearest illustration, because the high-level path never emits it and the validator pins its shape exactly.
[Listing](#lst:c26-sdpa-unit) names the four-or-five operand contract and the one attribute its parser reads.

```python
Unit "attn" {
  Type    = "SDPA"
  Bottom  = [ "q", "k", "v", "scale" ]   # optional 5th: additive mask
  Params  = { SubtractMax = true }        # ANECSDPALayerDesc byte 0x00
  OutputType = "Float16"
}
```

Listing: A hand-authored fused-attention Unit naming the native scaled dot-product attention descriptor in the network description. {#lst:c26-sdpa-unit}

The validator enforces the operand count with `4 or 5 bottoms must be present for SDPA`, requires the key and value to share a shape, and checks that the query times the transposed key contracts against the value.
The `SubtractMax` attribute is the single key the attention parser reads, and it defaults to false in the descriptor constructor, which is numerically wrong for softmax, so an authored attention Unit must set it true.
The optional fifth operand is an additive float16 mask broadcast over heads, zero on and under the diagonal and a large negative bias above it for the causal case.

At the backend-dialect level the same fused operation is one atom with a parametric contraction, the same form the matrix multiply uses, as [listing](#lst:c26-sdpa-atom) gives.

```mlir
anec.sdpa(%q, %k, %v, %scale)   // 4 or 5 bottoms; covers matmul + softmax + transpose
anec.matmul(%lhs, %rhs) { transpose_lhs, transpose_rhs }   // depth D must be 1 on both operands
```

Listing: The fused attention atom at the backend-dialect level, a single parametric contraction that covers the matrix-multiply, softmax, and transpose path. {#lst:c26-sdpa-atom}

The attention atom covers the matrix-multiply, softmax, and transpose path and is not gated behind the texture engine, which is why it runs on every family from the M1 onward.

## Authoring a hidden layer

A native layer reaches the silicon by naming its descriptor in the graph directly, then compiling and dispatching it as any other program, with the target chosen as the first family that runs the layer.
The graph names the native layer kind directly, and the compile step runs the layer's validator and assigns it to an engine, failing at compile time where the family gate rejects it, as [listing](#lst:c26-author) walks step by step.

```python
# Author a hidden layer via the bridge route: name the native layer kind and its parameters,
# cut it into the graph as a bridge node, then compile and confirm it lowers.

graph G:
    input q : [1, 8, 197, 64] fp16
    input k : [1, 8, 197, 64] fp16
    input v : [1, 8, 197, 64] fp16
    const scale

    # Describe the native layer directly as a bridge node, passed through untouched to the silicon.
    bridge_node attn:
        kind    = "SDPA"                      # the native descriptor name, not a decomposition
        inputs  = [ q, k, v, scale ]          # 4 operands; optional 5th mask is the causal case
        params  = { subtract_max = true }     # required: the default is false, wrong for softmax

    output attn

program P = compile(G, target = H13)   # runs the validator and assigns it to an engine
# Fused attention runs from the M1 onward (not texture-gated). A family-gated layer would fail here,
# at compile time, below its minimum family.
output = dispatch(P, q_data, k_data, v_data)  # confirm it lowers and runs on the target
```

Listing: Authoring a hidden layer through the bridge route, then compiling and dispatching to confirm it lowers and runs. {#lst:c26-author}

An authored layer is confirmed only by this compile-and-run on the target, not by the presence of its descriptor, since a gated layer fails the code generator below its minimum family.

## Arch-gated negatives

The same family gates as chapter 12 accept some authored layers on a later chip and reject them on the M1.
A native descriptor exists in the compiler binary on every target, but its validator has a minimum-family trait, and below that family the code generator rejects it.

The texture-engine samplers are the largest group: resize as a hardware sampler, crop-and-resize, grid resample, and the affine spatial transform are accepted from the A14 generation and rejected on the M1, where the compiler reports that the affine transform is not supported on this architecture.
The whole-tensor argument-minimum-and-maximum layer is gated to the A15 generation and rejected on the M1.
Range normalization is arch-gated and rejected on the M1.
The native circular state layers behind an on-device key-value cache are hard-gated on the M1 and reached there only through a shared resident buffer.
Authoring the Unit does not bypass the gate: the layer compiles where its family allows and fails at compile time where it does not, so the same network description targets the generation that first runs the layer and every generation above it.

A second class of rejection is not about family but about a layer that passes an earlier check and fails the code generator, the attested-is-not-reachable rule from chapter 4.
On the M1 the top-k, sort, and dynamic-slice validators are all callable, yet the code generator rejects sort and dynamic-slice and accepts top-k only outside a small forbidden parameter band.

## Two compile routes and their gates

The same operation can have different availability on the same chip depending on which route reaches it.
The conversion route from a high-level model is gated by a minimum-family trait on each operation, the floor the public conversion path checks.
The direct-authoring route is gated instead by the per-chip hardware-abstraction feature bytes the layer validators read.
The whole-tensor argument-minimum-and-maximum layer is the clearest example: the conversion route floors it above the M1, yet the direct-authoring route gates it on a feature byte that is set from the A13 onward, so it is rejected through conversion on the M1 and runs through direct authoring on the same chip.
Trigonometric sine and cosine have no direct-authoring bridge, so they stay conversion-only and reject on the M1; sort and top-k have a bridge but the code generator rejects it on the M1.

## Reference: the native layer classes reached by direct authoring

The native catalog is the set of layer kinds the engine silicon runs, each with a constructor and a validator in the compiler.
The classes [Table](#tbl:c26-classes) gives are the ones the conversion path does not emit and direct authoring reaches, each with the binding validator gate and the first family that runs it.

| Class | Native layers | Binding gate | First family |
| --- | --- | --- | --- |
| Fused attention | scaled dot-product attention | four or five operands; key and value share a shape; the subtract-maximum attribute must be set | M1, not texture-gated |
| Ranking and selection | top-k, sort, argument-minimum-and-maximum, whole-tensor argument form | index outputs returned float16-encoded; sort and top-k pass the validator and fail the M1 code generator | whole-tensor form A15; top-k and sort code-generated above the M1 |
| Spatial rearrangement | pixel-shuffle, pixel-unshuffle, space-and-channel, space-and-batch | per-axis integer factors that factor into {2, 3, 4, 8}; depth factor one | M1 |
| Normalization | range normalization, local response normalization, per-channel gain-offset control | range normalization arch-gated; gain-offset in static and runtime forms | range normalization above the M1; gain-offset M1 |
| Geometry and point cloud | template cross-correlation, three-vector cross product, furthest-point sampling, radius neighborhood, stereo cost volume | cross product requires interleave one and float16 operands; cross-correlation template depth one | M1 for the cross and correlation forms |
| Texture samplers | resize as a hardware sampler, crop-and-resize, grid resample, affine transform | the texture-engine feature byte | A14; rejected on the M1 |
| Data movement | re-strided input view, runtime-offset dynamic slice, tile, concatenate | input view must follow a reshape; dynamic slice passes the validator and fails the M1 code generator | M1 for the static movers |
| Streaming state | live state, ring-buffer reader and writer, tensor-to-buffer movers | the ring-buffer writer must connect to a live-state buffer; circular mode arch-gated | above the M1; reached on the M1 only through a shared resident buffer |

Table: The native layer classes direct authoring reaches; the full per-layer descriptor and parameter schema is in Appendix B. {#tbl:c26-classes}

## Reference: the descriptor lowering of a fused operation

[Table](#tbl:c26-lowering) lists the descriptor lowering of four representative atoms, each with its operand contract and the binding constraint its validator enforces.

| Atom | Operands | Binding constraint |
| --- | --- | --- |
| `sdpa` | four or five: query, key, value, scale, optional mask | key and value share a shape; query times the transposed key contracts against the value; softmax uses the family path |
| `matmul` | two | depth one on both operands; output channel equals the left channel; contraction parametric through the transpose flags |
| `gain_offset_control` | one, plus scale and bias | per-channel affine, height one; the fold target for a bias or activation |
| `layer_norm` | one, with gamma and beta folded | channel divisible by the group count; output type float; the grouped form is the same atom with a group count above one |

Table: The descriptor lowering of four representative atoms, with the operand contract and the binding constraint each validator enforces. {#tbl:c26-lowering}

## Programmable activation LUT

The named activations are not distinct hardware.
The compiler synthesizes each one, sigmoid, tanh, gelu, swish, and the rest, into a 33-knot piecewise-linear table over a fixed domain, and the engine evaluates that table.
The format is recovered byte for byte, 33 knot values at a fixed step and 32 inter-knot deltas behind a short header, and the operation set includes a custom-table opcode, `kZinIrNonLinearCustomLUT`, that runs an arbitrary pointwise function the same way.

The raw custom-table path is not reachable from a netplist.
The unit parser requires a saturation set and a version-specific set together, and a consistency check in the same routine then rejects their coexistence, so no authored table satisfies both and every attempt fails to compile.
The capability is real but sits below the user artifact: the firmware synthesizes the table from the program at load.

An arbitrary pointwise function still runs on the engine by composition.
A piecewise-linear curve over chosen knots is a `linear`, `relu`, `linear` chain, the same form the hardware table evaluates, so a small rectifier basis reproduces any knot table to fp16 exactly, demonstrated on a Gaussian bump that no named activation provides.
The engine's exposed model is thus a linear map, a pointwise nonlinearity, and a linear map: custom weights and any scalar function, but not a new arithmetic primitive, a custom reduction, or data-dependent control flow.
