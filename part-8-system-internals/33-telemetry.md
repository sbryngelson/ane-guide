# 33. Telemetry and hardware counters

> The engine has a hardware performance-counter block of twenty-four per-task-descriptor counters, master enable at `ANEProgramCreateArgs+0x6c`, and free-running firmware timestamp at MMIO `0x2_6b17_8000`.
> The block geometry and the timestamp are readable, but the per-task-descriptor counter values are not, because one kernel gate blocks them on the unentitled path.
> Forcing the stats mask non-zero turns a successful load into a rejected one, since the compiled program has no stats-descriptor section for the kernel to size.
> What remains readable is the whole-engine telemetry outside that gate: DRAM read and write bytes, engine energy in millijoules, and clock-state residency.

## Counter block geometry

[Table](#tbl:c33-counter-groups) gives the per-task-descriptor counter groups, the number of counters in each, and a representative counter name.

| Group | Counters in group | Representative counter |
| --- | --- | --- |
| Neural engine cycles | 6 | `kANE_NE_COMPUTE_CYCLES`, `kANE_NE_INPUT_STALL_CYCLES` |
| L2 (on-chip 2 MB) cycles | 4 | `kANE_L2_NOMINAL_CYCLES`, `kANE_L2_READ_STALL_CYCLES` |
| L2 processing-element cycles | 3 | `kANE_L2PE_COMPUTE_CYCLES` |
| Precision compute cycles | 2 | `kANE_FP16_CYCLES`, `kANE_INT8_CYCLES` |
| Kernel-manager stall | 1 | `kANE_KM_STALL_CYCLES` |
| Data-movement bytes | 7 | `kANE_DMA_READ_BYTES`, `kANE_L2_TO_NE_DATA` |
| Per-descriptor energy | 1 | `kANE_DPE_ENERGY` |

Table: The per-task-descriptor counter groups, with the count in each and a representative counter name. {#tbl:c33-counter-groups}

The per-task-descriptor counter namespace is twenty-four named counters, every one of them per task descriptor, grouped by engine block.
[Table](#tbl:c33-counter-namespace) gives the complete namespace, recovered from the counter-name accessor and grouped by engine block.

| Group | Counters |
| --- | --- |
| Neural engine cycles | `kANE_NE_NOMINAL_CYCLES`, `kANE_NE_COMPUTE_CYCLES`, `kANE_NE_THROTTLE_CYCLES`, `kANE_NE_INPUT_STALL_CYCLES`, `kANE_NE_OUTPUT_STALL_CYCLES`, `kANE_NE_KERNEL_STALL_CYCLES` |
| On-chip 2 MB memory cycles | `kANE_L2_NOMINAL_CYCLES`, `kANE_L2_THROTTLE_CYCLES`, `kANE_L2_READ_STALL_CYCLES`, `kANE_L2_WRITE_STALL_CYCLES` |
| L2 processing-element cycles | `kANE_L2PE_COMPUTE_CYCLES`, `kANE_L2PE_INPUT_STALL_CYCLES`, `kANE_L2PE_OUTPUT_STALL_CYCLES` |
| Precision compute cycles | `kANE_FP16_CYCLES`, `kANE_INT8_CYCLES` |
| Kernel-manager stall | `kANE_KM_STALL_CYCLES` |
| Data-movement bytes | `kANE_DMA_READ_BYTES`, `kANE_DMA_READWRITE_BYTES`, `kANE_AF_TO_KM_DATA`, `kANE_AF_TO_L2_DATA`, `kANE_L2_TO_AF_DATA`, `kANE_L2_TO_NE_DATA`, `kANE_NE_TO_L2_DATA` |
| Per-descriptor energy | `kANE_DPE_ENERGY` |

Table: The complete twenty-four-counter per-task-descriptor namespace, recovered from the counter-name accessor, grouped by engine block. {#tbl:c33-counter-namespace}

The byte counters track data movement along the activation-function to kernel-manager to L2 to neural-engine path, and one per-descriptor counter estimates energy from the digital-power estimator.
With this set a host can attribute, per scheduling unit, whether a task descriptor is compute-bound, input-stalled, output-blocked, or throttled, directly in hardware counters.
The master enable is a single field in the program-create argument struct.
It is the per-task-descriptor stats mask at `ANEProgramCreateArgs+0x6c`, a `u32` that is after the quality-of-service field and the packed boolean flags and before the memory-pool identifier, located in the argument fields of [Listing](#lst:c33-createargs).

```c
ANEProgramCreateArgs (offsets):
  +0x18..+0x57 : two SHA256-class hashes (model hash + key)
  +0x5c (u32)  : count (number of procedures)
  +0x64 (u32)  : qos                  = 21 (0x15)
  +0x68 (u32)  : packed bool flags    = 0x10
  +0x6c (u32)  : statsMask            <- the per-TD counter master enable
  +0x70 (u32)  : memoryPoolID         = 0
  +0x80        : program name "main_main__Op0_AneInference"
```

Listing: The program-create argument fields, with the per-task-descriptor stats-mask master enable at offset `+0x6c`. {#lst:c33-createargs}

The driver remaps a client-facing mask to a driver mask before it reaches this field.
The remap keeps only the low nibble: a mask of `0xffffffff` translates to a driver mask of `0x0`, that is, no collection, and `0xf` is the only fully-enabled translation.

$$\mathrm{driverMask}(m) = \begin{cases} \mathrm{remap}(m \bmod 16) & m < 16 \\ 0 & m \ge 16 \end{cases}$$

When the mask is non-zero the firmware writes each task descriptor's counter block into a shared stats buffer in DRAM.
The host decoder is built against the `sCAneStatsData` ABI version `0x0201`, distinct from the on-wire header magic `0x0101` the firmware writes into the buffer (chapter 29).
The buffer's required size is the sum of the stats header, event descriptors, and per-event records.
The host decodes it through a parser that walks a Group to Layer to task-descriptor hierarchy, where the leaf task-descriptor node holds the counter block.

## Free-running timestamp

A monotonic firmware timestamp underlies the whole telemetry surface.
The firmware reads it from a single free-running memory-mapped counter through the one-line helper of [Listing](#lst:c33-timebase), then stamps each trace-event record from that read.

```c
/* free-running engine timebase counter, read by the firmware helper @0x30988 */
#define ANE_TIMEBASE_COUNTER  0x26b178000ULL   /* MMIO 0x2_6b17_8000 */

static inline uint64_t ane_read_timebase(void) {
    return *(volatile uint64_t *)ANE_TIMEBASE_COUNTER;  /* ldr x0, [x8] ; ret */
}
```

Listing: The firmware helper that reads the free-running engine timebase counter. {#lst:c33-timebase}

Each firmware trace-event record has a `timeStamp` field alongside its task-descriptor identifier, network identifier, program identifier, process identifier, and task-queue, as [Listing](#lst:c33-trace-events) shows.

```text
[ANE_TM_EVENT_START]:          tid, nid, progId, procId, currTQ, timeStamp
[ANE_TM_EVENT_FINISH]:         tid, nid, progId, procId, currTQ, timeStamp
[ANE_EVENT_CONTEXT_SWITCH_IN]: tid, nid, prevTQ, progId, procId, currTQ, timeStamp
```

Listing: The fields of each firmware trace-event record. {#lst:c33-trace-events}

The host-visible clock residency runs in 24 MHz ticks, one tick every 41.67 ns, read out of the system-on-chip state-residency channels.
The timestamp is monotonic and survives a power-gate, since the firmware re-anchors it from the same free-running source rather than resetting it across a clock-state transition.
It is the basis for the per-dispatch wall-clock intervals that the signpost stream exposes, and it is readable with no entitlement beyond root.

## What the host can read and what it cannot

The block geometry and the timestamp are observable; the counter values are not.
The split follows the stats mask: the timestamp and the whole-engine channels do not route through it, and the per-task-descriptor counters do, as [Listing](#lst:c33-readable-blocked) contrasts.

```c
/* READABLE: no stats mask in the path */
uint64_t  t  = ane_read_timebase();                 /* free-running firmware timestamp */
int64_t   rd = ioreport_delta("AMC Stats|Perf Counters|ANE0 RD");  /* DRAM read bytes  */
int64_t   mj = ioreport_delta("Energy Model|-|ANE0");              /* engine energy, mJ */

/* BLOCKED: gated by the per-task-descriptor stats mask */
args.statsMask = 0xf;                  /* master enable, ANEProgramCreateArgs+0x6c     */
create = ANE_ProgramCreate(&args);     /* create -> 1                                  */
load   = ANE_ProgramLoad(create);      /* load   -> 0: initStatsBufferSection bails     */
perf   = read_perf_iosurface();        /* every byte 0: per-run output buffer is null  */
```

Listing: The readable timestamp and whole-engine channels contrasted with the blocked per-task-descriptor counter path. {#lst:c33-readable-blocked}

The values are blocked because the master enable never takes effect on the host path.
On the unentitled runtime path the runtime sets the stats mask to `0` below the model layer, so firmware collection is never armed and the per-run output buffer comes back null.
Forcing the mask non-zero through an in-process hook does not help: the program-create call then reaches the kernel routine of [Listing](#lst:c33-statsbuffer), which looks up the program's stats-descriptor section by name, reads its size field, and bails on a zero size.
The `aned` daemon is what zeroes the mask: it sets `statsMask=0` for a `coreAnalyticsClientType` of `ThirdPartyAppUsingANE`, so the mask is cleared by client type, and the host-side null check is the string `perfStatsIOSurface is NULL!`.

```armasm
initStatsBufferSection(ANEProgramCreateArgsOutput*, task*):
  ldr  w8, [x8, #0x28]     ; size of the stats-descriptor section
  str  w8, [x28]
  cbz  w8, bail            ; size 0 => return 0 (create fails)
  ...                      ; non-zero => kalloc + map the stats buffer
```

Listing: The kernel routine that bails when the stats-descriptor section size is zero, failing the create call. {#lst:c33-statsbuffer}

The compiled program has no stats-descriptor section, so the size is always zero and the kernel returns failure.
Forcing a non-zero mask thus turns a successful load into a rejected one (`create -> 1; load -> 0`), and no host-side primitive synthesizes the missing section.
One kernel gate thus blocks both the per-task-descriptor counters and the per-run output buffer: each depends on a stats-descriptor section that only an internal profiling-compile emits.

What remains readable is the whole-engine telemetry outside that gate, the channels of [Table](#tbl:c33-telemetry-channels) with their format, unit, and what each reads out.

| Channel | Format | Unit | Reads out |
| --- | --- | --- | --- |
| `AMC Stats \| Perf Counters \| ANE0 RD` | fmt=1 delta integer | B | DRAM read bytes |
| `AMC Stats \| Perf Counters \| ANE0 WR` | fmt=1 delta integer | B | DRAM write bytes |
| `AMC Stats \| Perf Counters \| ANE0 DCS RD` | fmt=1 delta integer | B | DRAM read bytes via the compression-subsystem path |
| `AMC Stats \| Perf Counters \| ANE0 DCS WR` | fmt=1 delta integer | B | DRAM write bytes via the compression-subsystem path |
| `Energy Model \| - \| ANE0` | fmt=1 delta integer | mJ | engine energy |
| `PMP \| AF BW \| ANE0 RD+WR` | fmt=2 residency | events | aggregate read-plus-write bandwidth events |
| `SoC Stats \| Cluster Power States \| ANE0` | fmt=2 residency | 24Mticks | clock-state residency |
| `SoC Stats \| Events \| SOC0_ANE_F1`, `SOC0_ANE_F2` | fmt=2 residency | 24Mticks | clock-domain frequency-point residency |
| `SoC Stats \| Events \| ANE0_ADCLK_TRIG`, `ANE0_DITHR_TRIG` | fmt=2 residency | 24Mticks | adaptive-clock and dither trigger ticks |
| `Interrupt Statistics \| ane0 0` | fmt=1 counters | counts, MATUs | first and second-level interrupt-handler count and time |

Table: The whole-engine telemetry channels readable on the unentitled path, with their format, unit, and what each reads out. {#tbl:c33-telemetry-channels}

The memory-controller per-agent byte counters report DRAM read and write bytes for the engine, both on the raw path and separately on the compression-subsystem path, and the energy model reports engine energy in millijoules.
The system-on-chip state channels report clock residency, frequency-point residency, and the adaptive-clock and dither trigger counts, and the per-map fabric arbiter reports the engine's bandwidth and clock-floor votes.
The interrupt statistics report the host kernel's cost of servicing engine completions, in counts and Mach Absolute Time Units, split into first-level and second-level handlers and isolated separately for the engine and its address-translation unit.
A storage coprocessor that an earlier reading mistook for the engine, exposing vector-lane read and write counters, is unrelated to engine telemetry; the bandwidth counter is the memory-controller `ANE0` channel and the power channel is the energy model.

The aggregate bandwidth channel resolves into per-channel histograms, the fabric channel `PMP0 / DCS BW / ANE L0` and `L1` carrying read and write distributions for the two DRAM read channels, read as State-residency histograms rather than plain integers.
The `SoC Stats / Events` channel also exposes the throttle-trigger family `ANE_THROTTLE_{SW,HW,PPT,DITHER,EXT}_TRIG` and `VDD_DRAM_VOLTAGE_CHANGE`, per-trigger software, hardware, peak-power, and dither throttle counts, all readable without entitlement.

The per-dispatch op lifecycle is also readable as the timestamped signpost stream of [Listing](#lst:c33-signposts), captured on the engine subsystem with no entitlement beyond root, which confirms the three-request structure of a dispatch without exposing any counter value.

```text
_ANEF_MODEL_EVALUATE                    one per host execute call
  _ANEF_MODEL_EVAL
  _ANEF_MODEL_EVAL_DRIVER_REQUEST       request 1 of the dispatch (Cast)
  _ANEF_MODEL_EVAL_DRIVER_REQUEST       request 2 of the dispatch (AneInference)
  _ANEF_MODEL_EVAL_DRIVER_REQUEST       request 3 of the dispatch (Cast)
  _ANEF_MODEL_EVAL_PERFCOUNTER_SAMPLE   the per-descriptor counter sample is reported here when armed
_ANEF_MODEL_COMPILE   _ANEF_MODEL_LOAD   _ANEF_MODEL_UNLOAD
_ANEF_IOSURFACES_MAP  _ANEF_INPUT_BUFFERS_READY  _ANEF_ENQUEUE_OUTPUT_SET
```

Listing: The client-side signpost intervals of one dispatch, with the three driver requests that correspond to the Cast, AneInference, and Cast program operations. {#lst:c33-signposts}

Each interval has a Mach-time begin stamp, and the three driver requests are the three program operations of a dispatch, one driver-to-firmware request each.
The firmware reports the blocked per-descriptor counter sample on the evaluate interval only when the stats mask is armed, so the signpost stream confirms the dispatch structure on live silicon while the counter values stay behind the same gate.

## Roofline and power figures from the readable channels

The measured roofline elsewhere in this guide rests on the readable whole-engine channels, not on the blocked per-descriptor counters.
Bandwidth comes from a delta on the memory-controller byte counters across a real workload.
A 96-layer matmul chain at batch 8192 moved 17.2 GB of reads and 16.9 GB of writes over 432 ms, which is between 79 and 90 GB/s, the engine's share of the unified memory.
A delta on the energy-model millijoule channel over the same run gives about 0.48 pJ/FLOP and about 2.3 W under sustained compute.

The compute roof and the dispatch floor are wall-clock measurements, timed against the firmware timestamp and the host clock rather than read from a counter.
The 2 MB working-set threshold is confirmed directly from the byte counters.
At batch 8192 the activation is exactly 2 MB, the counters show 426 MB of DRAM moved per dispatch, arithmetic intensity falls to 60 FLOP/byte, and throughput drops from 12 TFLOP/s to about 4.8 TFLOP/s.
The per-descriptor counters would attribute that drop to specific stall classes, the input-stall and L2-read-stall cycles, but the whole-engine byte and energy channels already locate the workload on the bandwidth slope, which is what the roofline needs.
