# 29. Firmware

> The engine controller runs a C++ application over a real-time kernel, distributed as an unencrypted preload executable. Every subsystem is a long-lived message-pumped task, and all kernel objects sit in fixed build-time pools.
> The execution loop accepts four command classes on the shared channel, three procedure-call variants plus the cache-request trigger, and rejects anything else as unsupported.
> The scheduler is fixed-priority with eight levels, no hardware watchdog, and a software deadline of about 2 seconds that aborts a stuck queue.
> Recovery is serialized behind a single command gate and rate-limited, so crashes arriving faster than the recovery cycle keep the channel not-ready.

The engine's controller runs a small real-time operating system, not bare metal.
A real-time kernel hosts an application that loads compiled programs, schedules their task descriptors, drives the multiply array and the data-movement engines, and reports faults back to the host.
That controller is CHINOOK, an embedded ARM core with eleven hardware thread contexts (`CHINOOK_CPU_IMPL_THID0` through `THID10`), its own level-two cache, and pipeline error-capture and power-down-save registers, so the firmware runs on a real multithreaded processor rather than a sequencer.

## Substrate

The firmware is a C++ application over a real-time kernel, distributed on the package as an unencrypted preload Mach-O wrapped in an Image4 container with payload tag `anef`.
The kernel layer provides tasks, a priority scheduler, semaphores, mutexes, message queues, two heap arenas, a fault handler, and a generic finite-state-machine framework.
The application layer above it provides the program loader, execution loop, task-descriptor driver, tensor-mover driver, address-translation manager, and power-control service.

Every subsystem is a long-lived task with its own stack, supplied by a message queue.
A task blocks on its queue, wakes on a posted message, runs that message to completion, and loops.
Each task has two queues: a task-context input queue and a separate interrupt-context queue.
An interrupt handler never runs subsystem logic in interrupt context.
It pushes a record into an interrupt buffer, posts the target task's interrupt queue, and returns, deferring the work to task context.
The named tasks include an idle task, task terminator, execution-loop pump, address-translation manager, tunable-register loader, load monitor, interrupt manager, call manager, server, and host remote-procedure-call daemon.

Each task is in one of the five scheduler states of [Table](#tbl:c29-task-states), recovered from the task-list dumper.

| State | Meaning |
| --- | --- |
| `RUNNABLE` | ready or running |
| `SUSPENDED` | explicitly parked |
| `WAITING` | blocked on a generic wait, such as a message queue |
| `SEMWAIT` | blocked specifically on a semaphore |
| `BLOCKED` | blocked |

Table: The five scheduler task states. {#tbl:c29-task-states}

The distinct semaphore-wait state, separate from the generic wait, is the heritage of the real-time executive model the primitives follow.
The message queue each task blocks on is a bounded-buffer control block holding a maximum message width, a ring depth, read and write pointers, a lock, and two counting semaphores.
The send side blocks when full and the receive side blocks when empty.
The kernel holds all its objects in fixed pools sized at build time: tasks, semaphores, mailboxes, queues, signals, timers, and mutexes each occupy a slot in a pre-sized table.
There is no unbounded dynamic task creation at run time, and the capacity of each pool is fixed and reportable as a total-and-available count.
A hard boot invariant ties the heap to the system reservation: the initial heap must exceed twice the reserved framework allocation, and the kernel refuses to boot if the arena is too small.

## Bring-up

The three-level kernel bring-up call chain of [Listing](#lst:c29-bringup) separates device attach from firmware bring-up.

```cpp
ANEHWDevice::start()              attach + provider + RTBuddy client + 7 MMIO banks + clocks
  ANEHWDevice::power_on_hardware()    clocks/power + MPM + resume
    ANEHWDevice::ANE_Init()             RTBuddy endpoints + scratch handshakes + first commands
```

Listing: The three-level kernel bring-up call chain. {#lst:c29-bringup}

Bring-up splits into two phases on distinct entry points.
`ANEHWDevice::start` attaches the device, wires the RTBuddy client, maps the register apertures, and enables clocks and power.
`ANEHWDevice::ANE_Init`, reached from `ANEHWDevice::power_on_hardware` on first power-on and on every wake, drives the firmware handshake.
This separation keeps device attach independent of the firmware boot that runs again after each sleep.

The kernel maps seven named register apertures, not six.
The six-bank count belongs to a different system-on-chip family and does not hold on the M1.
The apertures are constructed in order, each with a device-tree `reg` aperture index, and [Table](#tbl:c29-apertures) names the seven by role.

| Order | Bank | Role |
| --- | --- | --- |
| 1 | `ANE` | main control aperture, holding the scratch registers and the reset-vector register |
| 2 | `PS` | power-state |
| 3 | `PWGATE` | power-gate |
| 4 | `PTD` | page-table and translation-domain control |
| 5 | `ANEHAL1` | hardware-abstraction register aperture |
| 6 | `ANEHAL2` | hardware-abstraction register aperture |
| 7 | `ANEHAL3` | hardware-abstraction register aperture |

Table: The seven memory-mapped register apertures the kernel maps during `ANEHWDevice::start` on the M1. {#tbl:c29-apertures}

The first four apertures take a contiguous index base and three successive offsets.
The three `ANEHAL` aperture indices are read from per-family device fields, which is why the aperture count and indices differ across chips.

The kernel allocates the RTBuddy client during attach and stores it on the device object, with a name object retrieved from the `ANE` property and a separate mailbox object.
The endpoint itself comes up inside `ANE_Init` through `EnableRTBuddyEndpoints`, which holds the endpoint identifier literal `0xe400`.
The path first tries `ANE1Endpoint1`, and on failure cleans up and retries `ANEEndpoint1`.
The endpoint lookup resolves the named service, retrieves the endpoint object, and registers the inbound message handler, with the doorbell paths wired alongside.

`ANE_Init` then runs the firmware handshake in program order.
It quiesces pending work, brings up the `0xe400` endpoint, initializes the scratch mailbox registers, and selects a warm or cold boot by writing `1` or `0` to scratch register 7.
It reads the firmware boot address and writes it to the reset-vector register `rANE_H11_CHINOOK_IO_RVBAR`, then releases reset and polls.
Three scratch handshakes follow, observed by the kernel in the order [Table](#tbl:c29-handshakes) gives.

| Handshake | Register | Meaning |
| --- | --- | --- |
| 1 | scratch 7, first wake | the engine controller is alive after the reset-vector handoff |
| 2 | scratch 7, second wake | the channel-description table the firmware published is ready |
| 3 | scratch 3, plus a magic-number-1 check | final acknowledgement before interrupts are enabled |

Table: The three scratch-register handshakes the kernel waits on during firmware bring-up, in order. {#tbl:c29-handshakes}

Between the second and third handshakes the kernel walks the channel-description table the firmware published and resolves the named inter-process channels: `SHAREDMALLOC`, `TERMINAL`, `BUF_H2T`, `BUF_T2H`, `IO`, `IO_T2H`, `DEBUG`, and `DATA_CHAIN_H2T`.
If the second wake never arrives, the kernel powers the hardware off and on in a bounded retry loop and increments a failure count.

On the firmware side the reset-vector entry brings up the real-time kernel: it sets up the memory-management unit and the exception and interrupt stacks, then the kernel heap, subject to the boot invariant that the initial heap exceeds twice the framework reservation.
The firmware then checks its boot arguments and chip revision against the values the kernel passed, exchanges an inter-process-communication protocol version, and validates each ring buffer in the channel table against a ring-buffer version constant.
The firmware marks each control channel ready in turn and rejects commands addressed to a channel before it is ready.

Bring-up is ready to accept the first command when all of the following hold.

- The kernel has enabled the `0xe400` endpoint, mapped the seven apertures, and enabled clocks and power.
- The firmware has booted from the reset vector, brought up its heap, passed the chip-revision and protocol-version checks, published the channel table, and marked every control channel ready.
- The three scratch handshakes have completed in order.
- The kernel has logged that the engine controller is ready, enabled interrupts, and round-tripped the start command.

Only then does the kernel issue its first commands over the channel, in order: print-enable, start, the performance-monitoring-unit base set, a host-to-engine time synchronization, channel-property write, resource-information query, default-setting write, memory-cache power-on, and the shared-event-information initialization.
The resource-information query returns the engine count, cache-request limits, and maximum procedure count, and the default-setting write holds the context-switch latency threshold.

## Execution loop

The execution loop accepts the four command classes of [Listing](#lst:c29-exeloop-cmds) on the shared channel.

```c
union uCExeLoopSupportedCmd          ; insize <= sizeof(union uCExeLoopSupportedCmd)

[PROC CALL]              ProgId=%d  ProcId=%d  Proc=%d  Pri=%d
[PROC CALL WITH BARS]    + nbrOfCustomBars <= 32        ; custom buffer-access-register overrides
[PROC CALL WITH EVENTS]  + nbrOfSignalEvents in 1..16   ; wait/signal events
                         + (nbrOfWaitEvents + nbrOfSignalEvents) > 0
[CR TRIGGER]             cacheHandler 0x%llx            ; data-chaining cache-request trigger
```

Listing: The four execution-loop command classes accepted on the channel. {#lst:c29-exeloop-cmds}

The execution loop is the hot path that turns a host procedure-call command into work for the task-descriptor driver.
A host writes a command into the shared ring buffer.
The loop receives that command holding a program identifier, procedure identifier, procedure index, and priority.
It builds the request list and pushes the program's task-descriptor partitions onto a task queue; the task-descriptor driver runs them on the cores while the tensor-mover streams the operands.
The command is a fixed-layout record on the shared channel, and the loop logs each class as it accepts it.
A bounds check rejects any record larger than the supported-command union before the loop reads its fields.
The bar field is a buffer-access register, the buffer base and size binding a task descriptor references by index, and it logs as `bar[%d]: type=%d cfg=%d barId=%d value=0x%llx bufSize=%lld`.
The loop rejects an unknown opcode with `Cmd 0x%x is not supported through ExeLoop cmd channel` and drops the record.

The main command classes on this channel are a plain procedure call, procedure call holding buffer-base overrides, procedure call holding wait-and-signal events, and data-chaining cache-request trigger.
The dispatcher checks each command identifier on every event against a fixed accepted set and drops anything else as unsupported: the procedure-call family (`0x204`, `0x20c`, `0x211`, and `0x212`), the cache-request trigger (`0x209` with its path variant `0x20a`), the channel data-file load `0x2d`, and the back-channel and process-id controls (`0x404`, `0xff00`).
The complete decoded command set is in chapter 30.
The firmware drops and counts a procedure call addressed to a process that is not running rather than faulting it.

The procedure-call command structures hold the buffer-access registers and optional event and execute-order arrays under the fixed field limits of [Table](#tbl:c29-field-limits), recovered from the firmware assertion strings.

| Field | Limit |
| --- | --- |
| custom buffer-access registers | at most 32 |
| signal events | greater than 0 and at most 16 |
| wait events plus signal events | greater than 0 |
| task-descriptor partitions | at least 1 and below the per-procedure maximum |
| custom execute-order entries | at most 128 |
| output buffer sets | exactly 1 |
| event masks | exactly 1 |

Table: The field limits on a procedure-call command, recovered from the firmware assertion strings. {#tbl:c29-field-limits}

Each buffer-access register binds a buffer base and size that a task descriptor references by index, with at most 32 hardware register slots addressing the distinct buffer regions for one operation, and at most 16 input buffers per trigger.

The data-chaining trigger is the firmware mechanism behind resident state across dispatches.
A cache request chains one execution's output set onto the next execution's input set, so a value produced by one dispatch is available to the next without a copy back to the host.
The resident key-and-value cache and the resident optimizer state described in chapter 2 use this native chaining path.
The firmware can chain procedures, running several in sequence on the engine without returning to the host between them.

## Doorbell emit

[Listing](#lst:c29-doorbell-emit) gives the guarded doorbell-emit sequence that brackets the host-notify store with the window-gate bit clear and set.

```armasm
ring_doorbell(DoorBellReg, DoorBellBit):     ; host-notify site @0x4c890
  w0 = disable_irq()                         ; DAIFSet #0x2, make the quad atomic
  x8 = mrs S3_3_C15_C8_0
  x8 = x8 & ~(1 << 39)                        ; CLEAR bit 39: arm the doorbell window
       msr S3_3_C15_C8_0, x8
  str DoorBellBit -> [DoorBellReg]            ; *** the doorbell store (the interrupt) ***
  dsb sy ; isb                                ; force the store + status update to commit
  x8 = mrs S3_3_C15_C8_0
  if x8 & (1 << 1): fatal "Uncorrectable L2C error overflow"
  if x8 & (1 << 7): rc = 1, clear sticky {1,7} ; transaction rejected, retry
  x8 = mrs S3_3_C15_C8_0
  x8 = x8 | (1 << 39)                         ; SET bit 39: close the window
       msr S3_3_C15_C8_0, x8
  restore_irq(w0)
  return rc
```

Listing: The guarded doorbell-emit sequence, bracketing the host-notify store with the window-gate bit clear and set. {#lst:c29-doorbell-emit}

After staging a task-descriptor list the driver notifies the host by ringing a doorbell.
The engine-to-host interrupt is a single 32-bit memory-mapped store to a host-supplied register, bracketed by the windowed-store gate of chapter 27.
Bit 39 of the implementation-defined system register `S3_3_C15_C8_0` is cleared to arm the doorbell window, and the store is forwarded as a fabric-coherent transaction.
Bit 39 is set again to close the window.
The firmware masks interrupts across the four steps so the arm, store, sample, and disarm cannot be interrupted, and the two status bits sampled after the store hold the fabric response.
The same guarded sequence drives the engine-to-graphics-processor synchronization doorbell, which writes the value `1` to the fixed memory-mapped target `0x2_0646_8000` between the identical bit-39 clear and set.

The system register is not a single-bit gate.
At least six bit positions hold distinct meaning, recovered from the read, write, and sample patterns at the 28 access sites and decoded in [Table](#tbl:c29-fabric-reg).

| Bit | Role |
| --- | --- |
| 0 | fabric-path enable, cleared with bits 2 and 4 on the power-gate path |
| 1 | uncorrectable last-level-cache error, read-only and sticky |
| 2 | fabric-link status, must be 1 for the power-gate condition |
| 4 | fabric-link secondary status, must be 0 for the power-gate condition |
| 7 | transaction-reject status, clearable on a rejected doorbell |
| 39 | doorbell-window arm and disarm: clear opens the store window, set closes it |

Table: The decoded bit map of the implementation-defined fabric system register `S3_3_C15_C8_0`. {#tbl:c29-fabric-reg}

Clearing bit 39 opens a window in which a store to a fabric doorbell aperture is forwarded as a coherent doorbell transaction, and bits 1 and 7 are the fabric response status latched by the barrier after the store.
The power-gate path reads the register to sample bits 2 and 4 as fabric-link status before it clock-gates the engine, requiring bit 2 set and bit 4 clear, then clears bits 0, 2, and 4.

A completion event can fan out to up to three host doorbells.
The host doorbell target is a register-and-bit pair the host supplies in an endpoint descriptor and the engine stores resident, in three slots, so the shared-event path can notify the inference client, a cross-agent waiter, and a telemetry sink from one completion.

## Scheduler

[Table](#tbl:c29-scheduler) collects the firmware scheduler properties, covering its priority model, bands, preemption rules, watchdog, and software deadline.

| Scheduler property | Value |
| --- | --- |
| Priority model | fixed-priority, eight levels (0 through 7) |
| System band | levels 0 through 1 |
| Application band | levels 2 through 7 |
| Preemption within a task | none, run to completion |
| Preemption between tasks | by kernel thread priority |
| Hardware watchdog | none |
| Software deadline | about 2 seconds, queue-idle wait then abort |
| Task-queue identifiers | 1 through 255, identifier 0 reserved |

Table: The firmware scheduler properties. {#tbl:c29-scheduler}

The scheduler is fixed-priority with eight levels, numbered zero through seven.
The levels split into two bands: levels zero and one are reserved for the system, and levels two through seven are application priorities.
Preemption within a task does not occur: each message pump runs to completion, and preemption happens only between pump threads by kernel priority.
Two priority axes coexist.
The kernel thread priority orders the worker tasks, and the application job priority orders queued inference jobs onto the per-priority task queues.
They are distinct numbers on distinct objects.
No hardware watchdog panics the engine.
The firmware self-polices with a software deadline: a task queue that will not quiesce within about two seconds is force-aborted rather than left hung.

The submit path pushes work onto numbered task queues keyed by a network identifier in the range 1 through 255, with identifier 0 reserved.
The submit path guards the queue identifier below 8, polls the queue for vacancy, enables the queue, copies a 35-word task-descriptor register block into the queue's register file, and rings a submit doorbell.
Each queue occupies a fixed stride of `0x148` bytes in the engine register aperture, with the per-queue registers the submit path writes given in [Table](#tbl:c29-queue-regs).

| Offset within queue | Register |
| --- | --- |
| `+0x24000` | submit doorbell |
| `+0x24004` | size and count |
| `+0x24008` | priority and network identifier |
| `+0x25000` | queue enable |
| `+0x2500c` | queue status and vacancy |
| `+0x250a0` | buffer-access-register file |

Table: The per-queue register map, at the queue base plus the queue identifier times `0x148`, that the submit path writes to start hardware work. {#tbl:c29-queue-regs}

The eight application priorities map to a fixed in-firmware table of queue weights, `{1, 2, 3, 4, 5, 6, 30, 31}`, and scheduling is strict fixed-priority with per-priority credit and no preemption.
The scheduler bounds how many requests are in flight on the firmware at once and applies backpressure when it saturates, through `ANEScheduler::wakePendingRequestsQueueWithFWOverload` and `ANEScheduler::pendingRequestsWithFirmwareCount`, with a dedicated unwire thread that releases resources behind completed work.
The firmware-overload throttle is the mechanism behind the large-batch trainable-convolution stall described in chapter 19: a submission that overruns the firmware queue is throttled rather than dispatched.

## Control state machine

A finite-state machine drives the execution loop through the states of [Table](#tbl:c29-control-fsm), each described by what it does.

| State | What it does |
| --- | --- |
| `INIT` (0) | the start node, after the loop is constructed but before it is armed; no hardware work dispatches |
| `RUN` (1) | non-secure and runnable, the engine is powered and owned but nothing is in flight |
| `EXEC` (2) | a task-descriptor partition is in flight, non-secure; the state in which a handler is permitted to touch hardware |
| `EXEC-SECURE` (3) | in flight while the secure phase is asserted, the engine owned by the secure tenant |
| `PAUSE` | fully quiesced for a secure-mode boundary, rejects all events |

Table: The execution-loop control state machine, with what each state does. {#tbl:c29-control-fsm}

The framework stores no name for each state, so the meaning of each is reconstructed from the handlers that read the current state and from the events that drive it.
Four states have framework identifiers zero through three, and a fifth is a named transient crossed during a secure-mode switch.
Posted events drive the machine rather than direct state pokes.
The event alphabet is four entries: a command-dispatch event when a procedure call or trigger is accepted, drain event toward idle, enter-secure event, and exit-secure event.
A command accepted on the channel posts the dispatch event, which moves the machine from `RUN` to `EXEC` and pushes the task-descriptor list.
A request finishing returns the machine from `EXEC` to `RUN`.

Roughly seven independent handlers gate hardware work on the same test: the current state must be `EXEC` or `EXEC-SECURE` before the handler writes the memory-mapped registers or rings a doorbell.
A switch into or out of the secure phase quiesces the engine first, with the task queues disabled, and the machine passes through `PAUSE` between the two phases.
Any event delivered while in `PAUSE` is a firmware invariant violation and asserts.

Three smaller state machines are alongside the execution loop, each with explicit named states.
A two-state process machine marks each process slot idle or running.
A three-state cache-request machine moves a resident chain through available, in-use, and invalidating.
A two-state output-set machine marks a chained output buffer available or in-execution.

## Per-run statistics buffer

[Listing](#lst:c29-stats-path) traces the per-run statistics-buffer path, from the host size query and map through the firmware completion write.

```text
host: stats-buffer size-get            -> {hdr 0x28, evDesc 0x14, dbg 0x10, perEngine 0x20}
host: PreMap stats buffer {addr,size}  -> DART-map into the firmware aperture
        if addr == 0 or size == 0:  cbz skips the map; later writes have no destination
host: procedure call (inference)
firmware completion (CAneProgramManagerH11):
        per engine: copy the 16-byte counter quad from engine_block+0x30 into a descriptor
        header: magic 0x0101, per-call timing, per-call counter, descriptor-area size
host: reads validCount, logEvents, statEvents back out of the mapped buffer
```

Listing: The per-run statistics-buffer path, from the host size query and map through the firmware completion write. {#lst:c29-stats-path}

The firmware reports per-run timing and event counts into a statistics buffer the host supplies.
`CAneProgramManagerH11` packs the records on inference completion, but it never allocates the destination.
The host first queries the layout sizes, then maps a buffer through the address-translation manager, and the firmware writes into that mapping at completion.

The record begins with the `0x28`-byte header of [Table](#tbl:c29-stats-header).
The first two bytes are the magic value `0x0101`, a version stamp.

| Offset | Width | Field |
| --- | --- | --- |
| `+0x00` | 2 | magic `0x0101` |
| `+0x04` | 8 | per-call timing value, from the real-time-kernel timebase |
| `+0x0c` | 4 | word from the call object |
| `+0x10` | 8 | reserved, cleared |
| `+0x18` | 4 | word from the call object |
| `+0x1c` | 8 | per-call counter value |
| `+0x24` | 4 | descriptor-area byte length, the total size minus `0x28` |

Table: The `0x28`-byte statistics-buffer header `CAneProgramManagerH11` writes on inference completion. {#tbl:c29-stats-header}

After the header the firmware copies one descriptor per active engine, looping over up to 32 engines at a stride of `0x48` bytes in the engine table.
Each descriptor holds the engine identifier and type, call index, and engine index, followed by a 16-byte counter quad read from `engine_block+0x30`.
The size query returns four constants the host uses to size the buffer: a `0x28`-byte header, a `0x14`-byte per-event descriptor, a `0x10`-byte per-debug-event record, and a `0x20`-byte per-engine descriptor.
The per-engine values are software-tracked latency and event accumulators maintained by the firmware profiler, not raw register reads.
The buffer also holds running totals the host reads back: a valid-record count, log-event total, and stat-event total.

The read is gated by a host-side null check.
The map handler loads the host buffer address and size from the command, and a `cbz` on each skips the address-translation map when either is null, as [Listing](#lst:c29-stats-nullcheck) shows.

```armasm
PreMap stats buffer:
  x22 = host stats-buffer address     ; from the command
  cbz x22, skip                       ; address == 0 -> no map
  x23 = host stats-buffer size
  cbz x23, skip                       ; size == 0    -> no map
  ... DART-map {x22, x23} into the firmware aperture ...
```

Listing: The null check that skips the statistics-buffer map when the host supplies no address or size. {#lst:c29-stats-nullcheck}

When the host never maps a buffer, the mapped pointer stays null, the completion writer has no destination, and every counter value the host observes is null.
A separate mechanism reads the performance-monitoring-unit aperture at the fixed base `0x2_8e08_c000`.
Those reads are 32-bit power-status registers at a stride of eight bytes, up to five entries, whose low byte is a per-engine power state.
They drive power and frequency-scaling decisions and are never packed into the statistics buffer.
The kernel hardware performance-counter block is a third mechanism again, armed and drained by the kernel rather than the firmware, and the firmware does not touch it.

## Faults and recovery

A failed invariant routes into the kernel fault handler, which produces a post-mortem dump in a fixed order.
The dump records the exception class, fault-address and exception-syndrome registers, full general-purpose register file, cache-controller error registers, a restart count that persists across restarts, and a replay of the recent-command ring newest first.
The firmware keeps that command ring so a crash dump shows what the engine was running when it died.
The platform exception handler stringifies four exception classes for the dump: non-maskable interrupt, interrupt, fast interrupt, and synchronous abort.
The dump names the faulting task by name, marks whether the fault occurred in interrupt context rather than a task, gives the processor identifier, and prints a call stack.

The fault model below a true exception is abort-and-recover per task queue.
The firmware aborts a stuck queue through the engine driver's hardware-abort path, and an abort that does not clear within the two-second deadline escalates to a panic.
A dedicated timeout interrupt fires on a global stuck flag, logs an event record, and drives the abort.
When a per-queue abort cannot clear the fault, the firmware escalates to a host-coordinated reset: it takes a reset mutex and sends a reset notification holding the task-queue range the reset spans, telling the host to tear down and re-initialize.

The engine cannot write a filesystem, so it depends on the host to drain the dump.
After staging the dump the firmware blocks while the host pulls it section by section, then the host resets the channel and re-initializes the firmware to the ready handshake.

Firmware status is not a flat numeric enum.
A namespace of notification commands sent to the host holds it, alongside inline return values at the point of failure.
A generic per-channel error notification holds an opaque error index, a reset notification holds the task-queue range the reset spans, and a tile-sync error notification reports a data-movement error.

Faults divide into four dispositions.
A version-minor mismatch warns and continues.
A command addressed to a torn-down process or a wrong-state cache handle is dropped and counted.
A command failing integrity, section, or argument validation is rejected without running.
A failed invariant, a processor exception, or an uncorrectable cache error faults into the dump-and-reset path.

The recovery sequence is serialized behind a single command gate.
A fault stages a firmware dump, cancels outstanding commands, and re-boots the firmware to the ready handshake before the gate frees, so a fault has a recovery cost rather than completing instantly.
