# Interlude. Below the API: how the engine works

> A reader who only dispatches work to the engine can stop at the end of Part V; the back half is the mechanism beneath the programming surface, for reasoning about a numeric result, predicting an unmeasured chip, or debugging a compile failure.

Parts I through V describe the engine as a developer uses it: what it computes, how to reach it, how it performs, what it suits, and how to fit work to it.
That account stops at the programming surface.
The rest of the guide goes beneath that surface, to the chip and the system around it.

The back half is a reference for readers who need the mechanism rather than the interface.
It is written at a lower level than the front half, and shows the structures, registers, and command protocol directly.

[Table](#tbl:handoff-backhalf) shows how the four back-half parts and the back matter divide, each with its subject and the reader it serves.

```{=latex}
\renewcommand{\thetable}{I.\arabic{table}}\setcounter{table}{0}
```

| Part | Subject | For the reader who needs |
| --- | --- | --- |
| VI. The Silicon | The datapath and MAC geometry, and the memory hierarchy | The mechanism behind the numerics and the roofline |
| VII. The Toolchain and Encoding | The compiler, the program and container format, the HAL and its gates, the compression pipeline, and direct netplist authoring | The mechanism behind compilation, the program format, capability gating, and compression |
| VIII. System Internals | The kernel driver and its ABI, the address-translation unit, the firmware, the host-to-firmware command protocol, power and thermal, security and isolation, and telemetry | The path from a host call to the silicon, and what is and is not observable |
| IX. Cross-Silicon Reference | The full target set, per-family code generation, and the upper tier | To compile for and reason about chips other than the one in hand |
| Back matter | The methodology, the open questions, and the reference appendices | To trust the numbers and to find a specific value |

Table: The back-half parts and the back matter, each with its subject and the reader it serves. {#tbl:handoff-backhalf}

```{=latex}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
```
