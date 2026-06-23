# 19. Pitfalls and limits

> Five direct-path failure modes are silent or target-specific.
> An M1 last-axis-offset slice saturates above 4094, dynamic-weight convolution fails to compile above batch one, the saturating slice kernel is keyed to the M1 target, repeated failed compiles in quick succession can stall the shared compile service, and a true four-character-code image input is not reachable on the direct path.
> Each is a property of the private compiler or its data tables, so the defense is a build-time rule rather than a runtime check.
> Pace compiles after a failure, roughly 15 seconds apart, so a run of failed compiles does not stall the service.

A few of the engine's failure modes are silent, target-specific, or affect the shared compile service.
Five of them are what a developer building on the direct path encounters, and each has a concrete trigger and a stated avoidance.

## Slice saturation hazard on the M1

A last-axis slice with a nonzero start offset is not a free descriptor edit on the M1.
On that generation the offset copy routes through a crop-DMA that stores values in a fixed-point format with four fractional bits, an implied scale of 16.
The stored value is the input times 16, and the storage port clamps at the fp16 maximum, so any element whose magnitude exceeds the fp16 maximum divided by 16 overflows to infinity.

The threshold is exact and given by

$$V_{\max} = \frac{65504}{16} = 4094$$

A slice element at 4094 survives, an element at 4100 becomes infinity, and a control value of 60000 that never enters the offset copy stays finite.
The hazard is limited and easy to miss: it occurs only on the M1 family, only on a last-axis slice, only with a nonzero start offset, and only when a value on that axis can exceed 4094.
On this one axis a cross-chip route change turns a finite number into an infinity rather than moving a result by a unit in the last place.
The A14 generation saturates on the same slice; the A15 generation and every part above it take a plain fp16 route and do not saturate.

The consequence shows up in training.
A convolution weight gradient scaled up by a large loss-scale factor can push values past 4094 on the M1 and silently produce infinities.
Keep the values on a width-offset slice under the bound, which for the training case means capping the loss-scale on M1 targets, and prefer a zero start offset where the layout allows, since a zero last-axis begin avoids the offset-DMA path entirely.
[Listing](#lst:c19-slice-guard) checks at build time that a nonzero width-offset slice on the M1 stays under the 4094 saturation bound.

```python
V_MAX = 65504 / 16   # = 4094.0; the fp16 maximum divided by the Q.4 gain of 16

def slice_offset_is_safe(x, begin_w, on_m1):
    # A nonzero width-offset slice on the M1 routes through the times-16 crop-DMA.
    # A zero begin avoids the offset-DMA path entirely; on A15 and above the
    # route is plain fp16 and never saturates.
    if not on_m1 or begin_w == 0:
        return True
    return float(abs(x).max()) <= V_MAX   # 4094 survives, 4100 -> inf
```

Listing: A build-time check that a nonzero width-offset slice on the M1 stays under the 4094 saturation bound of the times-16 crop datapath. {#lst:c19-slice-guard}

## Dynamic-weight convolution above batch one

A convolution whose weight is a supplied runtime tensor rather than an embedded constant is an engine capability.
At a batch of one it compiles and runs and matches a reference convolution to a cosine of 1.0000.
At a batch of two or more the same program crashes the compiler service: the helper process dies mid-compile and the error reads as a lost helper application rather than a plain rejection of an unsupported operation.

The crash is specific to the dynamic-kernel path combined with a batch above one.
A constant-weight convolution at the same batch compiles without issue, and batches of 64 are routine on that path.
The reproduction is stable across fresh restarts of the compile daemon: batch one passes, batch two fails, every time.

The avoidance follows the capability boundary.
Single-image dynamic-weight convolution is usable for hypernetwork and per-sample-kernel inference where the batch is one.
Batched trainable convolution must take a different route: the dynamic-kernel path is closed above batch one, so a build that needs it should reject a dynamic-weight convolution at batch two or more before the program reaches the compiler.
[Listing](#lst:c19-dynweight-guard) rejects a dynamic-weight convolution at batch two or more before the program reaches the compiler service that the path crashes.

```python
def guard_dynamic_weight_conv(weight_is_runtime_tensor, batch_n):
    # A const-weight conv shares one kernel across the batch; a dynamic kernel
    # must be re-split per batch element, and that path is broken on the M1.
    # Reject at build time so the program never reaches the compiler service.
    if weight_is_runtime_tensor and batch_n >= 2:
        raise ValueError("dynamic-weight convolution is closed above batch one")
```

Listing: A build-time guard that rejects a dynamic-weight convolution at batch two or more before it reaches the compiler service that the path crashes. {#lst:c19-dynweight-guard}

## Saturating slice kernel is target-keyed

The saturating slice kernel is keyed to the target, not to the compiler build.
The same compiler binary emits a saturating slice kernel on the M1 target and a non-saturating one on the M5 target, because the slice lowering is specialized per hardware family rather than parameterized at runtime.

Inside the single binary the slice conversion is a C++ template instantiated once per family, so the same intermediate-language slice lowers through a different compiled converter for the M1 than for the M5.
Per-target hardware-abstraction fields then drive the DMA source path: a width granule used as both a divisor and a stride multiplier, and a set of patch-width clamps that resolve to a power-of-two granule.
On the M1 the family converter and the format selection pick the times-16 fixed-point DMA format for a last-axis-offset copy; on the M5 the same copy stays in plain fp16.
The trigger is exactly the M1 family plus a last-axis slice with a nonzero start offset, so the avoidance is the same as for the saturation hazard and the defense belongs in the per-target build path.

## Pacing compiles after a failure

A failed compile is not free of side effects on the shared compile service.
A compile that fails restarts the service, which takes a few seconds to come back, and failures that keep arriving faster than the service can restart between them keep it from making progress, so unrelated compiles slow down until the failures stop.
The effect is a function of how fast failures arrive, not how many occur: failures spaced out past the restart interval cause no degradation at all.
On detecting a failed compile, wait at least one restart interval, roughly 15 seconds, before the next compile, so a burst of failures cannot accumulate.
No hard failure-count cap is needed.

## A single slow compile is a separate failure mode

A slow compile is not the same failure as a run of failed compiles, and the back-off rule that handles repeated failures does not prevent it.
A trainable convolution compiled over a large batch has a compile cost that scales with the batch, because the image-to-column tensors grow with the batch and the tiling and partition cost grows with them.
On the M1 the same backward convolution graph compiles in about 1.9 seconds at a batch of 4, about 35 seconds at a batch of 64, about 79 seconds at a batch of 128, and never finishes at a batch of 1000.
Shrinking the spatial size does not help, since the cost tracks the batch and not the feature-map size.
The large-batch trainable-convolution stall is the firmware-overload throttle of chapter 29 in action: a submission that overruns the firmware command queue is throttled rather than dispatched, so the fix is to cap the batch or split the program, not to wait it out.
The defense is to mini-batch real training at a modest batch per step rather than compile one full-batch graph, and the same graph compiles without issue on the M5.

A second slow-compile path is a combinatorial subgraph search.
The compiler clusters a graph by a memoized search over cut points that collapses linear chains safely but is exponential in the width of parallel branches that reconverge into one consumer.
The search has no iteration cap, but an internal time budget abandons it and falls back to a cheaper partition, which on the M1 plateaus the worst measured fan-in at about 9.2 seconds rather than letting it run unbounded.
A single cluster with roughly twelve or more independent branches reconverging into one consumer is worth flagging at build time, since the 9-second per-cluster cost is itself worth avoiding and the budget bail is not guaranteed across compiler versions.

## Direct four-character-code image input

A direct image input that declares a true four-character-code interchange format, so the engine reads a camera or video surface with no host-side conversion, is a no-go on the unentitled direct path.
The capability is real and the syntax is fully reachable: the pixel-buffer input type, format enum, grammar, and type rules all parse and type-check, and the program reaches the backend.
The failure is at backend lowering: the program does not compile on the direct route.

The lowering needs setup that only the entitled model-input route supplies, the four-character-code format descriptor backed by a surface, which the unentitled path lacks.
The supported and terminal form on the unentitled path is the uint8 image input, which dequantizes in the graph and saves the host-side conversion at the input.
It produces output byte-identical to a host conversion, saves roughly two milliseconds per frame at 1080p, and avoids the unsupported operation entirely.

## A build-time pitfall check

The pitfalls share one defense: gate the graph before it reaches the compiler, keyed on the target family and the offending shape.
The estimate holds the per-target hazard flags, so a saturating slice, batched dynamic-weight convolution, or four-character-code input is rejected at build time rather than at the silicon.
[Listing](#lst:c19-pitfall-scan) scans a graph for all five known pitfalls, keyed on the target family and the offending shape, and stops the build before a flagged graph reaches the compiler.

```python
# Scan the graph for the known pitfalls and reject or rewrite before compiling, target = H13.
# Each hazard is keyed on the target family plus an offending shape.

target = H13
on_m1  = (family(target) is M1 or family(target) is M2)   # the saturating-slice generations
V_MAX  = 65504 / 16                                        # = 4094: fp16 max / 16 (Q.4 gain)

for each op in graph G:

    # Pitfall 1: last-axis slice saturation on the M1 (finite turns to infinity above 4094)
    if op is slice and last_axis_begin_offset(op) != 0 and on_m1:
        if max_abs_value_on_axis(op) > V_MAX:
            reject(op, "width-offset slice can saturate above 4094 on M1")
        # rewrite where possible: use a zero start offset to avoid the offset-DMA path entirely

    # Pitfall 2: dynamic-weight convolution above batch one (fails to compile)
    if op is conv and weight_is_runtime_tensor(op) and batch(op) >= 2:
        reject(op, "dynamic-weight conv needs batch one; use batch one or another route")

    # Pitfall 3: an unsupported op (validator may accept it, code generator rejects on this target)
    if not runs_on_target(op, target):
        reject(op, "op does not lower on this target; decompose or choose a later family")

    # Pitfall 4: true four-char-code image input (does not lower on the direct path, unentitled)
    if op is image_input and format(op) is fourcc:
        reject(op, "true 4CC input faults at lowering; use the uint8 in-graph image input instead")

if any op was rejected:
    stop before compiling                                  # never let a flagged graph compile

program P = compile(G, target)                             # reached only when the graph is clean

# Pitfall 5 is a rate effect, not a graph shape: after any compile FAILURE, back off at least one
# restart interval (~15 s) before the next compile, so a burst does not stall the service.
on compile_failure:
    wait(restart_interval)                                 # about 15 seconds
```

Listing: Scanning a graph for the five direct-path pitfalls and stopping the build before a flagged graph reaches the compiler. {#lst:c19-pitfall-scan}

## Symbolic shapes are not reachable on the direct path

The engine compiler holds a full symbolic-shape system: it accepts a dimension written as an unknown, type-checks two dynamic operands for matching symbolic expressions, and enforces affine constraints on symbolic dimensions.
The silicon can thus run a program that accepts a variable sequence length on one compile in principle.
The direct compile path does not reach it.
A relu over a tensor with an unknown dimension parses, but the compile fails with an unsupported-operation error, because the symbolic-shape machinery is behind the entitled model-input route that lowers enumerated or range shapes onto symbolic engine programs.

The direct path thus compiles one program per concrete shape, and there is no one-compile-many-shapes form.
For a variable-length workload the options are to pad to a fixed maximum length and compile once, to bucket a small set of lengths and dispatch the nearest with the compile cache making repeats free, or to recompile per length under the per-compile cost.
The same limit closes the native dynamic-slice and dynamic-offset key-value-cache primitive on the direct path, so the host manages cache state with fixed-shape windows.

## Reference: the five direct-path pitfalls

[Table](#tbl:c19-pitfalls) collects the five direct-path pitfalls, each with its symptom, its cause, and the workaround.

| Pitfall | Symptom | Cause | Workaround |
| --- | --- | --- | --- |
| Slice saturation (M1) | Last-axis-offset slice silently produces infinities above $4094$ | Offset copy routes through a Q.4 times-16 DMA that clamps at the fp16 maximum | Keep width-offset values under 4094; cap loss-scale on M1; prefer a zero start offset |
| Dynamic-weight conv at batch $\geq 2$ | Fails to compile | Dynamic-kernel path is closed above batch one | Use it only at batch one; reject batch $\geq 2$ at build time; take a different batched route |
| Target-keyed saturating slice kernel | Same source saturates on M1, non-saturating on M5 | Slice lowering is template-specialized per family; M1 picks the times-16 DMA format | Treat the slice route as M1-specific in the per-target build path |
| Failed compiles in quick succession | Unrelated compiles slow while failures keep arriving | Failures arriving faster than the shared service can restart between them | Pace compiles after a failure by at least one restart interval (~15 s) |
| Direct four-character-code image input | Does not lower on the direct path | The entitled surface descriptor the lowering needs is unavailable unentitled | Use the supported uint8 in-graph image input, the terminal unentitled form |

Table: The five direct-path pitfalls a developer encounters, each with its symptom, its cause, and the workaround. {#tbl:c19-pitfalls}

## Reference: the compile-time and capability limits

Beyond the five silent or service-level pitfalls, three further limits reject or stall a build rather than corrupt a result, which [table](#tbl:c19-compile-limits) gives with each symptom, cause, and workaround.

| Limit | Symptom | Cause | Workaround |
| --- | --- | --- | --- |
| Large-batch trainable convolution | Single compile runs for minutes and hangs on the M1 | Image-to-column tiling and partition cost scales with the batch | Mini-batch at a modest batch per step; do not compile one full-batch graph |
| Wide reconvergent fan-in | A single cluster compile climbs super-linearly and plateaus near 9 seconds | Memoized subgraph cut-search is exponential in parallel-branch width; an internal time budget bails | Flag a cluster with roughly twelve or more branches reconverging into one consumer |
| Symbolic shape on the direct path | A tensor with an unknown dimension parses but fails to compile | The symbolic-shape system is reachable only through the entitled model-input route | Compile per concrete shape; pad, bucket, or recompile per length |

Table: The compile-time and capability limits that reject or stall a build, distinct from the silent direct-path pitfalls above. {#tbl:c19-compile-limits}

## Reference: the firmware fault surface

When a fault does reach the silicon rather than the compiler, the firmware classifies it into one of a few responses, and the response decides whether a workload is rejected, silently dropped, or recovered through a reset.
[Table](#tbl:c19-fault-surface) gives each fault class, its detection mechanism, and whether the firmware rejects, drops, or recovers through a reset.

| Fault class | Mechanism | Firmware response |
| --- | --- | --- |
| Command integrity | Magic-word, checksum, padding, and 32-bit-address asserts on each command | Reject the command and return an error status to the host |
| Program section | Section bounds and overlap audit at load | Refuse to load with a sanity-check failure |
| Runtime mis-target | A call to a torn-down process or a wrong cache state | Drop and count, no error returned |
| Assertion or exception | A runtime-assert or a processor exception | Register dump, coredump, and reset |
| Uncorrectable cache error | The L2 controller error block and its overflow state | Coredump and reset, not a soft retry |
| Task-queue watchdog | A queue that will not quiesce within two seconds | Abort the queue |
| Host-side timeout | No completion within the deadline | Cancel outstanding commands and re-initialize the firmware |

Table: The firmware fault surface, with each fault class, its detection mechanism, and whether the firmware rejects, drops, or recovers through a reset. {#tbl:c19-fault-surface}

The reject-versus-drop split is explicit.
Integrity and section and argument validation reject a command and return an error, so the failure is visible to the host.
The firmware instead silently drops runtime mis-targeting, a call to a process that has been torn down or a trigger on a cache handle in the wrong state, and counts it in a soft-failure telemetry counter.
A misbehaving client thus shows up in the drop counters well before it faults, and a host-visible restart counter increments each time the firmware re-initializes after a fault.
