# 28. Address translation and the DART

> The engine reads and writes DRAM through its own IOMMU, the DART, which translates a device address into a physical page at a 16 KB granule across a 3.5 GiB aperture.
> A normal engine page is one 64-bit leaf word, the physical frame located unshifted with bit 63 set: $\mathrm{leaf} = \mathrm{phys} \mathbin{|} \mathtt{0x8000000000000000}$.
> Program-create re-bases each host-mapped buffer into a firmware aperture with high half `0x1bc4`, a second translation with no constant offset.
> The fault-capture registers cannot be read on the M1, because every function on the fault path ends in a kernel panic.

Chapter 21 covered the on-chip working set: the pool the engine reads and writes once data is resident.
The off-chip step that puts data there is the subject here.
The engine reads its inputs and writes its outputs directly from DRAM through an input-output memory management unit, the DART, which translates a device-visible address into a physical DRAM page before any DMA engine touches memory.

## DART and its address space

[Table](#tbl:c28-address-space) gives the page granule, aperture base and size, and active stream set read from the live device tree.

| Property | Value | Meaning |
| --- | --- | --- |
| page granule | `0x4000` = 16384 = 16 KB | the IOVA page; every buffer maps at this granularity |
| aperture base | `0x0` | IOVA window starts at zero |
| aperture size | `0xE0000000` = 3.5 GiB | the managed IOVA span, `0x0` to `0xE0000000` |
| active streams | `{0, 1, 2}` | the engine streams sharing one translation-table base; client isolation is separate, per-client contexts `mapper-ane0-iso1` through `iso7` |

Table: The page granule, aperture base and size, and active stream set, read from the live device tree. {#tbl:c28-address-space}

The DART is the engine's IOMMU.
A host tensor buffer never reaches the engine as a host virtual address.
The DART maps it into the engine's own address space, the device virtual address or IOVA, and the DMA engines issue reads and writes against IOVAs.
The DART holds the page tables that translate each IOVA back to a physical DRAM page, so a buffer physically scattered across DRAM appears IOVA-contiguous to the engine.
The DART instance serving the engine on the M1 is a single controller bound to the engine stream set, and its managed address window and granule come from the live device tree.
The controller is the device-tree node `dart-ane0` at physical base `0x85800000`, bound to the t6000-generation driver class, not the t8020 class.
It has four 16 KB register windows at `0x85800000`, `0x85810000`, `0x85820000`, and `0x85804000`.
[Table](#tbl:c28-dart-props) gives the live `dart-ane0` device-tree properties read from the controller node.

| Property | Raw | Decoded |
| --- | --- | --- |
| compatible | `dart,t6000` | the t6000-generation controller |
| page size | `0x00004000` | 16 KB IOVA page |
| stream-ID enable bitmap | `0x0000a001` | bits 0, 13, 15 |
| bypass bitmap | `0x0000a000` | bits 13, 15 |
| stream count | `0x10` | 16 stream slots |
| options | `0x25` | low byte of the live config word `0x80000025` |

Table: The live `dart-ane0` device-tree properties read from the controller node. {#tbl:c28-dart-props}

The host-side translation object is a three-level mapper chain: the controller driver owns a mapper nub, which owns the translation mapper the engine driver is handed.
That mapper holds the highest retain count of any mapper on the system, consistent with its role as the live tensor-mapping object.
The page granule is 16 KB, not the 4 KB of the host page tables.
The firmware validates the same value independently and rejects a wrong one with `MMU invalid page size: %x`, so the host maps and wires every engine DMA buffer at 16 KB granularity.
A sub-page buffer still consumes a full 16 KB IOVA page.
This 16 KB page is the coarsest of the three alignment scales on the M1: the 16-byte DMA-width granule of chapter 21 is below a 256-byte segment alignment, which is below the 16 KB DART page.

A two-level page-table walk, the L2 and L3 tables under the per-stream translation-table base, over the 16 KB page covers the 3.5 GiB span.
The per-stream translation-table base is in the controller's translation-table base register, measured live at `0x90022320` for the active streams: bit 31 marks the base valid, and the remaining field shifts left by 12 to the physical table base `0x10022320000`.
Streams 0, 1, and 2 share one table base, a single table for the active engine streams.
That base is in the same DRAM band as the measured leaf physical frames, so it points at the controller's own page-table memory.

Client isolation is separate from this engine-stream table.
The DART gives each client its own isolation context, the eight address-translation mappers `mapper-ane0-iso1` through `iso7` plus a base in the live IORegistry, so each client's buffers map into its own translation domain.
The secure exclave receives these contexts as the capabilities `ANEIsoID1` through `ID7`, confining each client's DMA to its own domain, the address-translation half of the exclave capability model in chapter 32.

The controller register layout is recovered from its capture routine, as the byte offsets into a stream's 16 KB register window in [Table](#tbl:c28-reg-offsets).

| Offset | Register |
| --- | --- |
| `+0x40` | error status: fault flag at bit 31, plus stream id and fault code |
| `+0x50`, `+0x54` | error address low and high, the faulting device address |
| `+0xfc` | enabled-streams global bitmap |
| `+0x100` + 4·sid | per-stream translation-control array |
| `+0x200` + 4·idx | per-stream translation-table base array |
| `+0x1000`, `+0x100c` | translation-buffer control and status |
| `+0x1020`, `+0x1028` | translation-buffer error registers |

Table: The controller register-offset map, recovered from the register-capture routine. {#tbl:c28-reg-offsets}

## Leaf page-table entry

The leaf entry that the DART stores per page is one 64-bit word.
For a normal engine data page it is the physical frame with the valid bit set, given by

$$\mathrm{leaf} = \mathrm{phys} \mathbin{|} \mathtt{0x8000000000000000}.$$

[Table](#tbl:c28-leaf-entry) gives the bit-field layout of that 64-bit leaf word.

| bits | field | value | note |
| --- | --- | --- | --- |
| 63 | valid / active | `1` on map, `0` on unmap | the only flag set for an engine page |
| 62 | aux protection class | `0` | set only when prot bit 3 is set; never reached on the engine path |
| 60 | aux protection class | `0` | set only when prot bit 4 is set; never reached |
| 59 | aux protection class | `0` | set only when prot bit 5 is set; never reached |
| 46:14 | physical frame | full physical address, unshifted | low 14 bits zero at the 16 KB granule |
| 13:0 | within-page offset | `0` | always zero at page granularity |

Table: The bit-field layout of the 64-bit leaf page-table entry the DART stores per page. {#tbl:c28-leaf-entry}

Bit 63 is the valid bit, set on map and cleared to the all-zero template on unmap.
The physical frame is the full 16 KB-aligned physical address located unshifted, so its low 14 bits are zero and the frame and the valid bit do not overlap.

The word holds the same shape for every live engine usage type.
The mapping software holds two software protection classes, read-write for inputs, weights, and intermediates, and a device-write class for outputs, yet both collapse to the identical leaf template of bit 63 alone.
The DART encodes access permission in the per-stream translation-control configuration, not in these high page-table bits.
Bits 62, 60, and 59 are aux-protection classes that the engine driver never sets, because its direction-to-protection mapper produces only the values $\{1, 2, 3\}$, whose bits 3, 4, and 5 are always zero.
Across 26178 measured leaf-map events none of those three bits was ever set.

The mapping path that produces this word stacks two translations.
The host pins the physical pages, then fills the leaf table by accumulating per-page segments and flushing them through the page-protection-layer write.
That write holds the 40-byte per-page segment structure of [Listing](#lst:c28-ppl-seg), from which the leaf word is assembled per page.

```c
struct ppl_iommu_seg {   /* 0x28 bytes, measured layout */
    uint64_t iova;       /* +0x00  device virtual address      */
    uint64_t phys;       /* +0x08  physical DRAM page           */
    uint64_t size;       /* +0x10  0x4000 (16 KB granule)       */
    uint64_t prot;       /* +0x18  3 = RW, 1 = device-write     */
    uint64_t reserved;   /* +0x20  0                            */
};
/* assembled leaf word, per page i:  leaf[i] = seg[i].phys | template */
/* template = 0x8000000000000000 (bit 63) on map, 0x0 on unmap        */
```

Listing: The per-page segment structure passed into the page-protection-layer write and how the leaf word is assembled from it. {#lst:c28-ppl-seg}

The leaf table itself is page-protection-layer memory.
The word above is the value the kernel hands to that layer to store, captured at the store register on a live serialized dispatch, not a read-back of the stored page.

The host maps each buffer under a usage code that names its role, held in the segment structure and recovered from the map call sites, with the codes and their protection classes in [Table](#tbl:c28-usage-codes).

| Code | Role | Protection class |
| --- | --- | --- |
| 1 | client input tensor | read-write (3) on the M1 |
| 2 | client output tensor | device-write (1) |
| 7 | intermediate buffer | read-write (3) |
| 8 | kernel and weights | read-write (3) |
| 9 | program text, task descriptors, working set | read-write (3) |
| 11 | program constants and scratch | read-write (3) |
| `0x1019` = 4121 | firmware power-on shared surface | firmware class |
| `0x101a` = 4122 | firmware resident heap | firmware class |

Table: The buffer-role usage codes and the protection class each maps under. {#tbl:c28-usage-codes}

A direction-to-protection mapper produces only the values 1, 2, and 3: a device-write output is class 1, a read-only input would be class 2, and read-write or no-direction is class 3.
On the M1 a configuration bit in the controller collapses the would-be read-only class 2 into read-write 3, so the controller maps read-only inputs read-write and enforces input protection upstream rather than in the leaf word.
The high `0x1000` bit on the firmware codes flags the firmware-owned shared class, distinct from the per-program client and kernel buffers.

## Host-to-firmware rebase boundary

The host-side translation resolves to a physical address for every buffer.
For a single matmul-with-activation load, every host-programmed buffer resolves to its named buffer role: input tensor, output tensor, weights, program text, constants, intermediate, working set, and two firmware shared surfaces.
[Table](#tbl:c28-nine-buffers) shows the nine buffers of that load, each with its usage code.

| Named buffer | Usage |
| --- | --- |
| input tensor | 1 |
| output tensor | 2 |
| weights | 8 |
| program text | 9 |
| constants | 11 |
| intermediate | 7 |
| working set | 9 |
| firmware shared surface | 4121 |
| firmware resident heap | 4122 |

Table: The nine buffers of one matmul-with-activation load, each shown with its usage code. {#tbl:c28-nine-buffers}

An in-place operation maps two distinct device addresses onto one physical page, a single shared memory descriptor that the runtime reads as input and overwrites as output.
The physical pages are scattered across the DRAM band and are not physically contiguous: the page table is what makes each buffer appear contiguous to the engine.

One further translation is past the host boundary.
The values the engine reads from its instruction stream are not host IOVAs.
At program-create the firmware re-bases each host-mapped buffer into its own resident aperture and patches that rebased address into the engine registers, so the engine-register operand values are in a firmware window with high half `0x1bc4`, not in the host IOVA band, shown in [Listing](#lst:c28-rebase).

```text
host IOVA band     -->    firmware DRAM-tile aperture     (no constant host-side offset)
```

Listing: The firmware rebase from the host device-address band into the resident firmware aperture. {#lst:c28-rebase}

The host IOVA and the firmware-aperture address are unequal and have no constant offset between them, so the rebase is a second translation rather than a fixed displacement.
The bridge that establishes it runs once at program-create and is cached.
On the cached dispatch path the host never re-emits the host-to-firmware pair, so the firmware rebase is the only translation below the host boundary there.

Three address spaces coexist for one load, recovered from the loaded program text and the page-table fill and named in [Table](#tbl:c28-address-spaces).

| Space | Address space | What it holds |
| --- | --- | --- |
| A | Host IOVA | host page table, fully resolved to physical |
| B | Engine-register | where the firmware writes the DMA-engine bases |
| C | DRAM-tile aperture | what streams: the weight and data tiles |

Table: The three coexisting address spaces for one load: the host device addresses, engine-register aperture, and firmware DRAM-tile aperture. {#tbl:c28-address-spaces}

The program text is a list of 44-byte register-write records, each pairing an engine-register address in space B with two DRAM-tile operand values in space C.
The engine-register address selects which data-movement engine to re-base: the weight-streaming sources, input-tile reader, and output-tile writer.
The operand values hold the high half `0x1bc4` of space C, the firmware-resident rebase of the host buffers, which is why a naive comparison of raw register addresses across loads fails: the aperture base moves with each program load.

### Firmware rebase arithmetic

The rebase that produces space C from a host IOVA is a pure linear translation recovered from the firmware itself.
The firmware validated-translate routine reads the descriptor IOVA, range-checks it against the mapped region, then computes the runtime address as

$$\mathrm{runtime} = \mathrm{mappedBase} + (\mathrm{IOVA} - \mathrm{dvaBase}).$$

There is one subtract of the IOVA-region base and one add of the firmware aperture base.
There is no shift, no mask, and no page rounding, which is why the host IOVA and the firmware-aperture address have no constant offset: the offset is the difference of two independent region bases that each move per load.

The three runtime quantities the formula needs are in the firmware aperture-config object at `engine+0xa000`, whose fields [Table](#tbl:c28-aperture-config) gives.

| Offset | Field | Meaning |
| --- | --- | --- |
| `+0xb98` | `dvaBase` | host / DART IOVA region base |
| `+0xba0` | `regionSize` | mapped region size |
| `+0xba8` | `mappedBase` | firmware runtime aperture base, the rebase target base |

Table: The firmware aperture-config fields at `engine+0xa000` that drive the rebase, populated at buffer-map time and absent from the firmware image. {#tbl:c28-aperture-config}

These three are firmware-runtime state set at buffer-map time, not constants in the binary; the image holds only the arithmetic and the field offsets.
The firmware then writes the rebased value into the engine DMA bar registers at `0x285c25020 + engine*0x148 + barId*4`, where `engine*0x148` is the per-engine MMIO stride and `barId*4` selects the word-indexed bar register.
Only the low 32 bits of the rebased address reach this MMIO path; a per-bar config flag gates wider bars onto a separate software descriptor table.
Both the rebase tail and the register-target prologue were emulated under unicorn and reproduced the formula and the `0x285c25020 + engine*0x148` register address exactly.

The firmware also clamps every address it programs into the DMA engines to a 32-bit ceiling, below the 3.5 GiB aperture.
Each device address the engine touches, text, weights, descriptors, intermediate, output, and chained buffers, must satisfy `addr >> 32 == 0`, asserted per buffer in the firmware.
The host allocator thus hands out only sub-4-GiB IOVAs, and the engine operates in the bottom 3.5 GiB of its address space.

Three distinct alignment scales govern an engine buffer, each coarser than the last, collected in [Table](#tbl:c28-alignment).

| Scale | Value | What it governs |
| --- | --- | --- |
| DMA width granule | 16 B | the data-movement quantum |
| segment alignment | 256 B | program-text and segment packing |
| page granule | 16 KB | the device-address page, allocation and wiring unit |

Table: The three alignment scales on the M1, from the finest DMA width granule to the coarsest page granule. {#tbl:c28-alignment}

The host may pre-map a buffer before the inference that uses it, through a pre-map command that establishes its device-address mapping ahead of the hot path.
The firmware keeps explicit buffer pools, tagging each mapped buffer with a pool identifier, and runs a buffer-recycle state machine that reuses output buffers across chained calls.
A per-process pool tracks outstanding requests with an in-flight count capped at 127 per request.

Under device-address pressure the kernel applies a least-recently-used eviction policy over its mappings, scoring each with `getDartBufferFreeUpScore` and freeing through the `FreeUpDart*` family.
A long-running process that maps more than the address window holds thus has mappings reclaimed rather than failed.

## Fault-capture registers

The DART captures a translation or protection fault into the single-shot register block of [Table](#tbl:c28-fault-registers), an error-status word, the faulting device address split low and high, and a translation-buffer status word.

| register | device offset | field |
| --- | --- | --- |
| error / status | `+0x40` | fault flag at bit 31, plus stream id and fault code |
| error address low | `+0x50` | faulting device address, low half |
| error address high | `+0x54` | faulting device address, high half |
| translation-buffer status | `+0x100c` | busy / error status |

Table: The DART fault-capture register block, giving each register device offset and the field it holds. {#tbl:c28-fault-registers}

These registers cannot be read safely on the M1.
The capture routine writes the block into a DRAM snapshot only on the fault path, and on this controller that path is unconditionally a kernel panic.
Every function on the fault path ends in a direct call to the kernel panic routine, and control flow reaches it: the driver implements a DART fault as a `REQUIRE(...)` assertion, which panics the machine rather than returning the captured registers.
A read-only probe that reads the snapshot at the function boundary is built and validated against the disassembly, but it cannot fire without a fault, and any fault panics the box.
The faulting-address and status values are thus stated here as the structure of the block, not as measured values, since obtaining them on this hardware would require panicking the machine.
The register layout above is recovered from the capture routine; the field decode of the words a contained fault would return follows the published controller field layout.

### `IODARTErrorInfo` fault descriptor

Above the hardware DART-side capture is a software fault descriptor, `IODARTErrorInfo`, that the kernel `t6000dart` core constructs from the DART fault MMIO and hands to each registered consumer.
This descriptor is the structure the driver fault callback reads and logs, and it is a kernel-wide ABI: the sibling DART consumers `AppleAVD` and `AVE_DART` read the identical layout at the identical offsets, which confirms it is not an engine-private struct.
Most fields are object pointers whose stringifier the callback invokes; [Table](#tbl:c28-errorinfo) gives the byte offsets into the descriptor.

| Offset | Field | Type | Meaning |
| --- | --- | --- | --- |
| `+0x00` | `Type` | string | fault type, also the event-id header |
| `+0x08` | `HwClass` | string | hardware fault class |
| `+0x10` | `HwError` | string | hardware error code |
| `+0x18` | `HwStatus` | `u32` | raw DART fault-status word |
| `+0x20` | `IsWrite` | bool | `0` read fault, `1` write fault |
| `+0x28` | `SID` | `u32` | DART stream id of the faulting agent |
| `+0x30` | `Address` | `u32` | faulting IOVA |
| `+0x40` | `TTBRIndex` | `u32` | translation-table base index |
| `+0x50` | `L2Index` | `u32` | page-table walk L2 index |
| `+0x58` | `L3Index` | `u32` | page-table walk L3 index |
| `+0x90` | `AXI_ID[0..3]` | 4 × 8-byte slot | AXI master / transaction ids, each printed as a 32-bit value |

Table: The `IODARTErrorInfo` software fault descriptor, the shared kernel ABI the `t6000dart` core hands to its registered fault consumers. {#tbl:c28-errorinfo}

The four slots at `+0x90` are the `AXI_ID[0..3]` array, the four fault descriptors a prior decode reported.
The per-fault metadata is the scalar set above: `Address` at `+0x30` localizes the faulting page, `SID` at `+0x28` names the stream.
`Type`, `HwClass`, and `HwError` at `+0x00`, `+0x08`, and `+0x10` give the fault taxonomy, and `IsWrite` at `+0x20` holds the read-write bit.
The walk indices `TTBRIndex`, `L2Index`, and `L3Index` at `+0x40`, `+0x50`, and `+0x58` localize the failing page-table entry.

One register the callback reads is not part of the descriptor: it reads an engine status word directly off its own device object at `[engine+0xe028]`.
The callback treats the fault as benign and returns early when `(status | 0x80) == 0xa0`, that is when the low seven bits equal `0x20` and bit 7 is a don't-care.
Any other value is a real fault.
The same word gates the engine clock elsewhere in the driver, where `0xa0` is the powered-idle state.

The ANE kext does not raise the panic.
The kext fault path returns and, on a real fault, sets a sticky latch, dumps the shared-memory allocation table and firmware debug state, then parks for 250 ms awaiting external recovery.
The machine-halting panic is in the kernel `t6000dart` core, where the mapping and page-table-walk fault detection is a `REQUIRE` assertion that calls `panic()` directly.
That substrate panic fires before, or instead of, the kext recovery park.
