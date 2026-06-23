# 8. Entitlement boundary

> The direct route reaches the engine's useful compute, and four features are behind the framework loader or an entitlement: three-dimensional convolution, native stateful types, bf16 program input and output, and flexible or symbolic shapes.
> Each gated feature passes an earlier layer and fails at the one that runs the work, and none of the four runs on the engine on the M1 under either route.
> One hard limit is below all of this: a hand-built or self-compiled program is rejected at load with error `0xe00002e2`, so the reachable surface is everything the system daemon's compiler will accept.

The direct route reaches the engine through the execution runtime, below the model framework, with no placement planner in the path.
The sanctioned route reaches it through the model framework, which adds a segmenter, model container, and set of loader features the direct route does not have.
The direct route reaches the engine's useful compute; four features are behind the framework loader or behind an entitlement.
This boundary is about technical reachability alone; whether a path is supported by Apple or permitted for distribution is a separate and stricter matter, treated in the front-matter status note.

## What the direct route reaches

The operations that on-device perception and numerics networks are built from compile and run on the direct route without any entitlement.
Two-dimensional convolution and its transpose, matrix multiply, fused attention, the normalizations, the activation tables, pooling, elementwise arithmetic, and the data-movement operations all run, bounded only by the per-chip limits of chapter 4.
Compressed weights stream where the family supports it and reconstruct to fp16 where it does not.
The compiler accepts these operations and the runtime dispatches them, and an entitlement gates none of them.
An entitlement gates a separate set of loader-tier features instead, not the compute the direct route already reaches.

## Features that are gated

Four features are attested in the hardware capability tables or recognized by the compiler frontend, yet do not run on the direct route.
Each is gated, and the gate is at a different layer in each case, as [table](#tbl:c8-gated) gives with what gates each and how the direct and framework routes behave.

| Feature | Gated by | Direct path | Entitled/framework path |
| --- | --- | --- | --- |
| Three-dimensional convolution | Missing backend lowering on every device mask | Frontend recognizes it, lowering fails with a not-implemented rejection | Segmenter places it on the central or graphics processor, not the engine |
| Native stateful types (state, ring buffer) | Absent counter-and-event direct-memory-access engine in the M1 descriptor | Compile fails; reserved operation classes do not lower | Model container wires the stateful types; engine still lacks the primitive on the M1 |
| bf16 program input and output | Serialization layer and datapath; bf16 is not among the eleven program-I/O dtype codes | Compile fails with an unsupported-dtype rejection | Framework accepts the bf16 array, then casts to fp16 before the engine segment |
| Flexible and symbolic shapes | Runtime path, not the compiler; the symbolic-shape gate is on for the M1 | A symbolic dimension parses, then fails to lower | Compiles a small set of fixed shapes ahead of time and dispatches the nearest |

Table: Features blocked on the direct path, what gates each, and how the direct and framework routes behave. {#tbl:c8-gated}

Each gated feature passes an earlier layer and fails at the one that runs the work.
[Listing](#lst:c8-attempts) shows where each is admitted and where it stops, for three-dimensional convolution, a bf16 program output, and a symbolic shape.

```python
# 3D convolution: the frontend recognizes a static-weight 3D conv (it enforces
# "3D Convolution does not support dynamic weights"); lowering then fails on every mask.
conv(input, weight, kernel=[d, h, w])
  -> frontend: accepted
  -> lowering: "Not implemented": Some ops are not supported on any of the specified backends

# bf16 program input/output: not among the eleven program-I/O dtype codes.
# ANECIRDataType = {0 int4, 1 uint8, 2 int8, 3 fp16, 4 fp32, 5 int16, 6 uint16,
#                   7 int32, 8 uint32, 9 int64, 10 uint64}   # bf16 absent
function_output dtype = bf16
  -> compile: "Unsupported function output dtype bf16"   # no code to serialize it

# symbolic shape: a "?" dimension parses, then will not lower on the direct path.
tensor<fp16, [1, ?, 64]>
  -> parse: accepted (the symbolic variable s0 binds)
  -> lowering: "Not implemented": Some ops are not supported on any of the specified backends
```

Listing: Where each gated feature is admitted at one layer and rejected at the next, for three-dimensional convolution, a bf16 program output, and a symbolic shape. {#lst:c8-attempts}

A missing backend lowering gates three-dimensional convolution.
The hardware capability tables advertise a kernel-depth dimension, and the frontend recognizes a three-dimensional convolution with a static weight, but lowering then fails on every device mask with a not-implemented rejection.
No backend on the M1 has a code-generation path for the three-dimensional convolution operation class.
The sanctioned route does not run it on the engine either: its segmenter places the operation on the central processor or the graphics processor instead, so on this silicon there is no on-engine path to it for any caller.

An absent hardware primitive gates native stateful types.
The model framework documents the stateful type and flexible input shapes at developer.apple.com/documentation/coreml, and this account corrects the impression that those documented capabilities run on the engine directly [AppleCoreML].
The compiler reserves the operation classes for resident state and ring buffers, the operation table even naming a ring-buffer writer and reader and the counter-and-event direct-memory-access opcodes, but the in-place update needs that counter-and-event engine, and it is stubbed out of the M1 hardware descriptor.
Every register setter for it, the source and destination base addresses, shape and stride setters, atomic-operation and counter-mode setters, and wait-event address and value setters, has a body that asserts the engine is unsupported on this architecture.
A program that declares native resident state fails the compile, and this gate is stronger than an entitlement: the silicon path does not contain the engine, so no entitlement, property, or compiler option synthesizes it.
A resident cache on the M1 is built instead from a shared buffer kept live across dispatches, the documented zero-copy path for this generation.
The counter-and-event engine arrives on a later generation, so native resident state is a per-family feature there, not an M1 one.

The bf16 program input and output type is gated by the serialization layer and by the datapath.
The dtype enumeration that declares a program's input and output types has eleven codes, covering fp16, fp32, and the integer widths, and bf16 is not among them, so a program that declares a bf16 input or output fails the compile with an unsupported-dtype rejection.
The framework accepts a bf16 array at its boundary, but it casts to fp16 before the engine segment, so the engine runs no bf16 datapath under the sanctioned route either.
The fp16 datapath of chapter 3 already holds the accumulation precision a bf16 declaration would otherwise imply.

Flexible and symbolic shapes stop at the runtime path, not at the compiler.
The compiler has a full symbolic-shape system, with its master gate on for the M1, yet the direct runtime path will not drive it: a program with a symbolic dimension parses and then fails to lower.
The sanctioned route reaches flexible shapes by compiling a small set of fixed shapes ahead of time and dispatching the nearest, with a dynamic remainder on the central processor.
That is bucketed fixed-shape specialization, not one symbolic program on the engine, and the direct route matches it by padding to a fixed maximum or compiling a small set of length buckets, with the compile cache making a repeated shape free.

## Image-input boundary

A fifth feature is at the boundary in a sharper form: direct image-format input, where a camera or video surface is sent to the engine in its native four-character-code pixel format with no host-side conversion to fp16.
The operation that performs this on-engine conversion, `pixel_buffer_to_tensor`, is present in the intermediate-representation parser on the direct route and parses without an unknown-operation rejection.
Its input is a distinct surface type, `pixel_buffer<format, shape, bytes_per_row>`, and the format token is a `FMT_*` enumeration that maps onto the per-chip interchange-format table, so the grammar, format enumeration, and type rules are all reachable and solvable on the direct route.

The boundary is at lowering.
On the direct route the intermediate program parses and type-checks and reaches the engine backend, but it does not lower there, so the conversion cannot be compiled on the direct route.
The entitled framework route supplies the image-input descriptor and the surface setup that the lowering needs, so direct four-character-code input is a framework-route feature on the M1.
The terminal direct-route form is an on-engine integer-to-fp16 dequantization of a plain byte input, which avoids the unsupported operation and still removes the host-side conversion.

## What the boundary means concretely

The choice between the routes is a choice of loader and convenience, not of reachable compute.
Vision, encoders, on-device numerics, and training of the supported operation set are built from the operations the direct route compiles and dispatches, none of which the gated set touches.
The entitled path adds a bounded list: a segmenter that places arbitrary models across three devices with fallback, model container with flexible-shape and stateful wiring, and loader-tier feature set.
It does not add a different engine: the four gated features are absent on the M1 under both routes, or are matched by a direct-route construction that does the same work.

## Load-time signature check

One hard limit is below the compute and defines the whole access model: a hand-built or self-compiled program cannot be loaded onto the engine.
The kernel driver verifies every submitted program before it reaches the firmware, by a corecrypto signature check over the program bytes and a trustcache check on the backing file's vnode, and rejects a program that fails either check at load with error `0xe00002e2`.
The only program the kernel will load is one the system daemon compiled and signed in place, so a caller cannot author a network binary by hand and submit it.
This is why the direct route does not build a loadable binary: it authors a network in the intermediate representation and hands it to the daemon, which compiles and signs it on the caller's behalf, and the caller then drives the resulting signed program.
The reachable surface is thus everything the daemon's compiler will accept, not everything the engine could in principle run.

## Unentitled dispatch path

The kernel driver rejects a binary the client signs itself at load with `0xe00002e2`, so the daemon's compile-and-sign step is not optional.
The path has three steps.
First, the client authors the network as intermediate language rather than a loadable binary, since a self-authored binary never passes the load check and the client does not produce the final program itself.
Second, it hands that intermediate language to the system daemon, the one process able to sign: the daemon compiles and signs it in place, and the returned program carries the signature and trustcache trust the kernel load check requires.
Third, the client drives the returned signed program over the kernel interface directly, where the corecrypto signature and trustcache checks pass because it is daemon-signed and the program loads onto the engine.

## Reference: the boundary constants

[Table](#tbl:c8-constants) collects the constants of the entitlement boundary on the M1.

| Constant | Value |
| --- | --- |
| Program-load rejection code | `0xe00002e2` |
| Program-I/O dtype codes (`ANECIRDataType`) | 11 codes: 0 int4, 1 uint8, 2 int8, 3 fp16, 4 fp32, 5 int16, 6 uint16, 7 int32, 8 uint32, 9 int64, 10 uint64 |
| bf16 program input or output | absent from the 11 codes; compile rejected |
| Three-dimensional convolution | no backend lowering on any device mask |
| Native stateful types | counter-and-event direct-memory-access engine stubbed out of the M1 descriptor |
| Flexible and symbolic shapes | parse and bind, then fail to lower on the direct runtime path |
| Load-time signature checks | corecrypto signature over program bytes plus trustcache vnode-trust check |
| Entitlement-rejection return code | `0xe00002c7` (`kIOReturnUnsupported`) |
| Native counter-and-event engine | every register setter asserts unsupported on this architecture |
| Direct image-format input | `pixel_buffer_to_tensor` parses, then does not lower on the direct route |

Table: The constants of the entitlement boundary, M1/H13. {#tbl:c8-constants}

The entitlement check the direct route never trips returns a distinct code from the program-load rejection.
The program-load check returns `0xe00002e2` from the signature and trustcache check, while the entitlement gate that guards the higher inference-tier features returns `0xe00002c7`, `kIOReturnUnsupported`, the code a client gets when a gated feature's entitlement is absent.
The four gated features above fail upstream of either kernel code, inside the compiler or the firmware descriptor, which is why a host-side entitlement moves none of them.
