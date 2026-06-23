# Appendix D. Glossary

> This appendix defines the acronyms, proper nouns, and symbols the guide uses across all nine parts.
> Read the four core facts first, then the term table, then the family and silicon map.

The term [table](#tbl:apd-terms) is the reference; the notes after it record the core facts that the rest of the guide depends on.

## Core facts

Read these four corrections before the table; several entries below depend on them.
The M5 base part is H17, specifically the `h17s` compiler target, not H16s; H16s and H17s are separate targets distinguished only by the generation-tag byte.
The multiply-accumulate datapath uses a single wide accumulator of the fp32 class, supplied by radix-4 fp16-rounded input tiles; the accumulator width is fixed hardware on every device and is never a per-chip parameter, so it is not a recursive fp16 reduction tree.
On M1/H13 two weight-compression forms stream natively, the int4 palette (int4-LUT) and the sparse form whose mask and values have at least 50 percent zeros; int8 and blockwise-affine weights fold into the descriptor on that generation and only stream natively from later families.
The firmware task-queue notification identifier (NID) is 8-bit, taking values 1 through 255.

## Terms

| Term | Definition |
| --- | --- |
| A11 through A18, M1 through M5 | Apple system-on-chip marketing names, each with an ANE of a specific H-generation, related by $M(n) = H(n+12)$. |
| AFPP | The on-device firmware program container: a three-level big-endian FourCC package, `ANEH` then `ANEP` then sections. |
| `aned` | The system ANE broker daemon at `/usr/libexec/aned` that holds the IOKit access gate, so every unentitled client reaches the engine by sending it a request. |
| `aneuserd` | The per-user sibling of `aned`, the other holder of the IOKit access gate. |
| ANEC, `anec.*` | The compiler backend intermediate-language dialect of 97 operations, the target of front-end lowering and the input to task-descriptor codegen. |
| ANECCompile | The backend compile entry point that direct netplist authoring supplies rather than bypasses. |
| ANECompiler | The single compiler binary that lowers the front IR to the backend IR and then to task descriptors for every target, so one host can construct any of the 28 targets' hardware-abstraction blobs. |
| ANECompilerService | The out-of-process compile service; repeated failed compiles in quick succession can stall it, so pace compiles after a failure by about 15 seconds, covered in Part V. |
| ANEServices | The user-space framework layer beneath the runtime that marshals requests into IOKit calls. |
| AppleH11ANEInterface | The kernel driver for the engine, decompiled at version 9.511.3, that holds the IOKit class hierarchy and the user client. |
| ASC_CHINOOK | The firmware's internal chip codename string for the H13 ANE coprocessor. |
| bridge ops | Hidden backend layer kinds reached by direct netplist authoring, such as fused attention, fused rank, and fused rearrange, each paired with one frontend bridge module, described in Part II and cataloged in Appendix B. |
| CCDMA | The cross-chip and cross-engine DMA and event-sync engine; on M1/H13 it is folded, and it is present natively on A15 and later and on M5, where it enables resident state. |
| CSNE, `CSNE_CMD_*` | The host-to-firmware command protocol, where `CSNE_CMD_*` are the numeric command opcodes the host mailbox issues to the firmware. |
| DART | The ANE's IOMMU, a 16 KB-page, 3.5 GiB-window unit that maps host physical RAM into the engine's device address space. |
| DPE | Dynamic Power Estimation: a firmware activity-counter power estimate calibrated by 10 device-tree coefficients and bounded by the peak-power ceiling. |
| DVA | Device Virtual Address: the address the engine issues, resolved by DART to physical RAM, used interchangeably with IOVA. |
| dispatch floor | The fixed per-dispatch latency, about 0.23 ms on the M1 anchor and a fitted 0.11 ms on the M5, below which a kernel cannot run regardless of its size, covered in Part III. |
| `.e5` | The compiled-program dispatch-layer container whose size tracks the segment and dispatch count rather than the operation count. |
| e5rt | The E5 runtime, the C API that the frontend uses to load and stream programs and the unentitled reachable surface, which still relays to `aned` underneath. |
| EIR, NitroIR | The runtime's lowered IR, a Lisp-style S-expression node tree serialized on disk, whose pivot type is the fp16 `ndarray<half>`. |
| Espresso | Apple's cross-backend neural-network runtime and scheduler that hosts the E5 execution engine and the cost-model placement segmenter. |
| ExeLoop | The firmware's main control execution loop and finite-state machine that fetches and dispatches task descriptors through the RUN, IDLE, and EXEC states. |
| fp16 | Half-precision IEEE float, the engine's native compute and storage type, with a maximum finite magnitude of 65504. |
| fp16 accumulator | A shorthand tag for the MAC numeric behavior: a single wide accumulator of the fp32 class supplied by radix-4 fp16-rounded input tiles, where the only quantization is fp16 input and partial rounding and the fp16 output grid. |
| fp8 (E4M3, E5M2) | The 8-bit float weight and activation datapath, present only on H18, which decodes to fp16 before the MAC. |
| generation-tag | The hardware-abstraction blob's generation byte at offset `0x0`, holding the hex H-number, the decisive discriminator between near-identical targets such as H16s and H17s. |
| GOC, DynamicGOC | Generate-Output-Channels, the dynamic unit that generates the output-channel-group kernel tiles from a runtime weight, present from M1 onward. |
| HAL | The Hardware Abstraction Layer: the per-target scalar and byte blob that data-drives nearly all per-family behavior and is the source of truth for capability, limit, and cost, detailed in Part IX. |
| `.hwx` | The fully lowered hardware-executable container, the counterpart to the `.e5`. |
| int4-LUT, palette weights | Palettized 4-bit weight compression, one of the two formats that stream natively on M1/H13 (alongside the sparse form) at about 2.37 times, where int8 and blockwise-affine instead fold into the descriptor. |
| IOVA | IO Virtual Address: the device virtual address DART produces from host physical RAM, on a 16 KB page over a 3.5 GiB window. |
| KMEM | The on-chip working and scratch buffer the task descriptor sizes for weights and tiles, gated at 64 KB at legalization and the basis of the working-set threshold. |
| KV-cache (resident) | An on-device key and value cache that stays resident across dispatches for decode, built on M1 with `share_buffer` rather than native state, covered in Part VIII. |
| LUT | Lookup table, used both for piecewise-linear activation approximation and for the palette of int4 weight compression. |
| MAC | Multiply-accumulate, the compute primitive whose datapath is an fp16 multiply, radix-4 fp16-rounded input tiles, and one wide accumulator of the fp32 class. |
| MIL | The Model Intermediate Language, the front IR in single-assignment form that the compiler segments and lowers to the backend IR. |
| MLComputePlan | The model-framework introspection surface that reports per-operation device assignment and a cost weight, the readable view of the placement segmenter. |
| `.mlmodelc` | The compiled model bundle that pairs the runtime net, shapes, weights, and the `.hwx`. |
| multi-die, AllReduce, AllGather | The multi-die collective-communication layer present on the multi-die H14 through H18 Max and Ultra-class dies, not a base-class feature. |
| NE core | A neural-engine compute core; the per-family count decodes from the HAL as base 4, then 8, 16, 32, or 64 by suffix, described in Part IX. |
| NID | The firmware task-queue notification identifier owned by the state machine, 8-bit and taking values 1 through 255. |
| OCG | The Output-Channel Group, the compiler's output-channel tiling unit sized to the accumulator file, where a larger group means fewer DMA re-bases. |
| OC/cycle | The per-cycle output-channel throughput of the MAC array, the roofline unit in the cost model. |
| Path-A | Direct netplist authoring, hand-writing the backend netplist to reach hidden layer kinds, which supplies `ANECCompile` and so cannot reach a lowering the backend rejects, covered in Part II. |
| palettization | Weight compression that maps each weight to a lookup-table index, the int4 form of which streams natively on M1/H13. |
| power domains | The five independently gated power domains of the H13 ANE, by which the engine modulates power through the number it energizes. |
| PPL | Page Protection Layer: the kernel page-table protection layer through which the ANE's DART leaf writes go. |
| PPT | The peak-power ceiling and throttle under which the Dynamic Power Estimation values are bounded. |
| `pushTDList` | The firmware function that hands a task descriptor to the hardware and re-enters with an already-built descriptor for resident chains. |
| roofline | The performance bound that takes the smaller of compute-limited and bandwidth-limited rates as a function of arithmetic intensity, the basis of the cost model in Part III. |
| RTBuddy | Apple's coprocessor real-time-OS runtime framework, the substrate the ANE firmware app runs on. |
| RTKit | The real-time-OS substrate beneath the ANE firmware, providing the task and thread model and the synchronization primitives. |
| `share_buffer` | The runtime primitive that aliases an output buffer to an input buffer after compile, giving a zero-copy resident cache without native state. |
| slice ×16 saturation | An H13 codegen defect in which a slice with a nonzero last-axis begin lowers to a scaled kernel that silently sends values above 4094, which is $65504 / 16$, to infinity; H13-only and clean on H17. |
| SoC T-number | The SoC part number, such as T8103 for the M1 base and T8142 for the M5 base, which maps to an ANE H-generation through the board-type sequence. |
| sparse-binary, SparseFmt | The sparse-weight compute format, where the weight has a binary sparsity mask; the mask-and-values form streams natively on M1/H13, while the packed sparse-binary palette-index form is absent from the M1 version-5 descriptor and present from A15 and M5. |
| SPTM | Secure Page Table Monitor: the kernel monitor that, with the Page Protection Layer, governs page-table edits and physical-frame ownership. |
| stack layers | The top-to-bottom software path from the frontend through the runtime, the framework, `aned`, ANEServices, IOKit, and firmware to silicon, in Part VIII. |
| styx | The firmware and chip codename for the M1/H13 ANE. |
| TD, task descriptor | The hardware work unit the firmware loads and the engine executes: a register-image descriptor of DMA sub-blocks and framing, emitted per generation from the compiler's descriptor struct, detailed in Part VII. |
| TileDMA, KernelDMA | The conv datapath DMA engines: the kernel source streams weight coefficients, the tile source streams input activation tiles, and the tile destination writes outputs. |
| TM, Tensor-Mover | The firmware tile-manager driver that moves tiles and drives the texture layers. |
| TQ, task queue | The firmware queue the state machine enqueues a program's task-descriptor partitions into. |
| wide accumulator | The fp32-class running sum of the MAC, which holds small addends rather than dropping them, so a sum of representable terms stays near-exact, covered in Part III. |
| Winograd | The Winograd fast-convolution transform the compiler can emit for small kernels, trading multiplies for transforms. |
| Zin, ZinIr, ZinMir | The compiler's internal class namespaces: the IR-object layer, the mid-IR build layer, and the task-descriptor codegen layer. |

Table: The acronyms, proper nouns, and symbols used across the guide with their definitions. {#tbl:apd-terms}

## Family and silicon map

The relation $M(n) = H(n+12)$ is anchored at both ends, with the live M1 reporting `h13g` and the M5 cost-model trees decompiled on the M5 host reporting `H17C` and `H17S`.
The compiler-family index drives operation legality, and the per-target HAL drives codegen, limits, and cost.
[Table](#tbl:apd-family-map) maps each marketing name to its ANE generation, architecture string, compiler family, generation-tag, and core counts; Part IX gives the full table.

| Marketing name | ANE H-gen | OS arch string | Compiler family | generation-tag | NE cores by suffix |
| --- | --- | --- | --- | --- | --- |
| A13, M1 | H13 | `h13`, `h13g` | A13 | `0x0d` | 4, 8 (g) |
| A14, M2 | H14 | `h14`, `h14g`, `h14c` | A14 | `0x0e` | 4, 8, 32 |
| A15, M3 | H15 | `h15`, `h15g`, `h15c` | A15 | `0x0f` | 4, 8, 32 |
| A16, M4 | H16 | `h16`, `h16s` | A16 | `0x10` | 4, 8, 16, 32 |
| A17, M5 | H17 | `h17`, `h17s` | A17 | `0x11` | 4, 8, 16 (M5), 32, 64 |
| A18 | H18 | `h18` | A18 | `0x12` | 4 |

Table: The Apple chip marketing names mapped to ANE generation, architecture string, compiler family, generation-tag, and core counts. {#tbl:apd-family-map}

The M5 base is the `h17` runtime arch and the `H17s` compiler target, the 16-core variant, not H16s.
The suffixes `g`, `s`, `c`, and `d` decode to NE-core counts of 8, 16, 32, and 64 from a single HAL field.
A17 and A18 add no new operation capabilities over A16 and scale only the core count; the A13 to A16 jump was the last capability expansion.
The fp8 datapath is H18 only.
