# Open questions

> Open questions fall into three kinds: questions another chip or an on-silicon probe would answer, questions the sanctioned entitlement would answer, and questions that are impossible to answer because the data does not exist or the silicon leaves no observable trace.
> Each item gives why it is open and what would close it.

The engine is decoded from the host call down to the register writes, and everything that exists as bytes or strings has been resolved into human-readable form.
Two silicon points are measured directly: M1/H13 and M5/H17s.
M5/H17s confirmed all ten cross-silicon predictions on device.
The M3/H15 generation and the upper tier above H17s are decompile-derived, not measured.
M5/H17s closes none of the upper-tier items, because it is a 16-core H17s part rather than the H17d 64-core ceiling, H18-gated fp8 runtime, or Ultra multi-die collective.

## Why an item is open

Each open item falls into one of three kinds, which set the three tables below.
A chip-measurement item has its structure decoded but needs another generation, the upper tier, or an on-silicon probe to read the realized values.
An entitlement item is attested in the hardware-abstraction layer or the intermediate language but is reachable only through the sanctioned model path.
An impossible item cannot be resolved at all: either the data does not exist, or the behavior is irreducible silicon that leaves no trace in any result and no artifact to decode.

## Ledger

Three tables sort the open items by how each could be resolved.
[Table](#tbl:openq-chip) gives the items another chip or an on-silicon probe would resolve, each with its decoded structure and the measurement that would close it.

| Open item | Why it is open | What would close it |
| --- | --- | --- |
| Live values of the single-shot fault and translation-fault capture registers | The registers are located (four fault descriptors and a status word at engine+0xe028) and the ANE-side handler has a benign branch, but the IODART substrate panics on a real fault, so the live values are uncaptured | A fault path that recovers instead of panicking the IODART substrate |
| Realized A15 / M3 / H15 silicon behavior | The dedicated A15 compiler branch is decoded (its cost table, its 45-operation family set, the full H15 targets, the YUV420 input it alone accepts), only the realized numbers need the part | On-chip measurement of an H15 part |
| Realized upper-tier behavior: 64-core H17d, H18-gated fp8 datapath, Ultra multi-die collective | The encodings are decoded (the core-count parameter, the fp8 e4m3 and e5m2 convert-and-quantize path, the Ultra device-mesh collective) and the upper tier adds only cores over A16, no new operations, but none of it materializes without the part | An H17d, H18, or Ultra part |
| Set behavior of the family-gated abstraction-layer fields | The fields are named and the silicon-capability subset is complete, their set behavior needs later silicon | A part where the gate is on, for example the A18 or H17-plus FIFO-mode field |
| Per-state frequency and voltage of the operating-point sequence | The engine has no local DVFS and is driven by the SoC power controller, the firmware seven-step credit sequence is recovered, the per-state frequency and voltage stay behind the opaque power-management base | A power-management-side probe |

Table: Open items a further on-silicon measurement would resolve: the structure is decoded, the realized values need the part or the probe. {#tbl:openq-chip}

[Table](#tbl:openq-entitlement) gives the items the sanctioned entitlement would resolve, attested in the binaries but gated off the direct path.

| Open item | Why it is open | What would close it |
| --- | --- | --- |
| Live per-run firmware performance-counter values | The read path and its gate are decoded (the `aned` daemon clears the stats mask for a third-party client, and a host-side null check skips the buffer on the unentitled path), only the live values need the entitled output buffer | The entitled path that supplies the output buffer |
| Entitled-only features: 3-D convolution, native state and ring buffer, bf16 program input and output, flexible shapes | Attested in the hardware-abstraction layer but unreachable on the direct path | The sanctioned model path that can reach them |

Table: Open items the sanctioned entitlement would resolve, attested but gated off the direct path. {#tbl:openq-entitlement}

[Table](#tbl:openq-impossible) gives the items that cannot be resolved: the data does not exist, or the behavior is irreducible silicon that leaves no trace in any result.

| Open item | Why it cannot be resolved |
| --- | --- |
| Gate-level MAC and adder-tree wavefront skew | The output is order-independent, so no result reveals the per-cycle wiring, which sits below the timing floor; only Apple's register-transfer netlist would show it |
| Exact fp16 partial-sum rounding order within one MAC reduction | The wide accumulator makes the output bit-identical for any summation order, so the internal sequence leaves no trace; only the register-transfer netlist would show it |
| No flat numeric error or status enumeration | Status is held by notification names and positional indices, not a stored enum; the flat enumeration does not exist |
| The execution-loop state labels | The five states are reconstructed from handler behavior; no name table exists in firmware |

Table: Open items that cannot be resolved externally: irreducible silicon behavior, or data that does not exist. {#tbl:openq-impossible}

## Form of the residual

The impossible items are few: the gate-level reduction wavefront and the internal fp16 rounding order are irreducible silicon, and the flat status enumeration and the execution-loop state labels do not exist as stored data.
Everything else is decoded in structure and waits only on a part, a probe, or the entitled path: the chip-measurement items the tables above list, and the entitlement items attested in the binaries but gated off the direct path.
Static analysis closed the firmware address rebase that once sat at the host and firmware boundary, and the M5 measurement confirmed the cross-silicon model without reaching any of the parts the hardware items need.
