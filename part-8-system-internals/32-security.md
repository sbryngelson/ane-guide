# 32. Security and isolation

> The trust boundary for a submitted program is the kernel driver, not the firmware: the driver checks the program signature, on-disk trustcache, and client's code-signing identity before any work reaches the engine, which then runs only structural bounds checks.
> Secure mode is a working transition that gives one tenant the quiesced, power-cycled engine; exclave mode is inert `mov w0, #0; ret` selector stubs on the M1 but the live, capability-scoped execution substrate on the M5.
> The engine is a shared timing and occupancy oracle: a confirmed cross-process side channel leaks a co-tenant's presence and a coarse duty cycle at roughly 20 to 50 bit/s, but never a value, weight, or input.

The kernel driver enforces the engine's security model almost entirely above the firmware. It vets every program before the work reaches the engine, secure mode isolates one tenant in time rather than in hardware, and the one confidentiality gap is a timing side channel rather than a data leak.

## Secure and exclave mode

[Table](#tbl:c32-mechanisms) gives the secure-mode and exclave mechanisms with their status on the M1 and the supporting evidence.

| Mechanism | M1 status | M5 status | Evidence |
| --- | --- | --- | --- |
| Program signature check | live, kernel-side | | corecrypto link, signature symbol |
| Vnode trustcache check | live, kernel-side | | vnode trust symbol |
| Firmware program check | structural only, no crypto | | bounds and overlap asserts |
| Secure-mode FSM | live, working | | non-secure to secure transition |
| Secure isolation | temporal, power-cycled | temporal plus exclave capability partition | quiesce plus power-cycle plus pause |
| Exclave firmware switch | dormant, stubbed | live | `SwitchExclaveMode not supported` on the M1 |
| Exclave host binding | present, real bodies, unbound | bound, proxy active | proxy IOService, recovery FSM |
| Exclave selector ABI | inert stubs on M1 | live implementations | `mov w0, #0; ret` on the M1 |

Table: The secure-mode and exclave mechanisms, with M1 and M5 status and supporting evidence. The M5 was measured with System Integrity Protection enabled. {#tbl:c32-mechanisms}

The firmware has no cryptography of its own, and the secure-boot chain authenticates it externally.
The trust boundary for user-submitted programs is the kernel driver, which checks the program signature and validates the backing file against the platform trustcache before the work ever reaches the firmware; the firmware then performs only structural bounds checks.

The kernel driver enforces three independent checks on a submitted program before it reaches the firmware.
The first is a cryptographic signature over the compiled program bytes, run by `AneMachoSignatureCheck` against the platform code-signing trust root, for which the driver links the system cryptography library.
The second is a vnode trust check, `aneVnodeTrustVerification`, that validates the on-disk model file against the platform trustcache and ties a mapped buffer back to its backing file to defeat a map-then-swap race.
The third binds a resident program to the submitting client's code-signing identity through `GetTeamIdAndCodeSigningId` and `hasSameCodeSigningId`, so one client cannot attach to another client's resident program or cache.
Once the kernel clears the program and loads its sections, the firmware checker runs only bounds, overlap, and type checks under a verification banner, with no hashes, and trusts that the kernel has already vetted the signature and provenance.

Secure mode is a real, working transition that gives one tenant exclusive ownership of the quiesced engine.
The state machine moves between a non-secure and a secure phase under the commands `CSNE_CMD_SECURE_MODE_START`, `STOP`, and `RESUME_TRANSITION`, with a firmware-to-host `CSNE_CMD_SECURE_MODE_EVENT` and a host acknowledgement.
A boolean `aneSecurePhase` records which side the engine is on, and the transition runs in four steps.
The engine first reaches readiness with no pending work.
A quiesce command drains in-flight work and disables the task queues.
The firmware then power-cycles the engine block across the boundary.
The execution loop finally enters a paused state, with the reverse path returning it to a running state.
While secure, the firmware silently drops and counts non-secure cache-request triggers.
The boundary power-cycles the block and drains every task queue and DMA channel, so a secure tenant starts from a reset engine and neither side observes the other's residue.
This is temporal single-engine isolation, not a hardware partition, since the device exposes one physical engine.

The engine firmware has no digital-rights-management or content-decryption code of its own.
A full string sweep finds no FairPlay, Widevine, PlayReady, or content-decryption module; the only `CDM` token in the image is `CDMediaBusManager`, a Common-DMA media-bus endpoint manager over the inter-processor ring buffers, not a Content Decryption Module.
Secure-mode exclusivity and the host trust chain handle protected media rather than an in-engine content module.
The kernel admits a protected workload only after it verifies the program signature and trustcache and puts the engine in secure mode so no other tenant shares it while protected buffers are resident.

Exclave mode is compiled into the loaded binary but dormant on the M1.
The firmware commands `CSNE_CMD_EXCLAVE_MODE_START` and `STOP` exist, but the M1 handler returns `SwitchExclaveMode not supported`, and the support is a runtime capability bit rather than a compiled-out feature.
The host binding runs ahead of the firmware side.
The proxy IOService, secure-to-exclave plumbing, secure-processor handoff, and firmware-recovery state machine all exist as real function bodies in the loaded driver, fifty-one exclave symbols in all.
What is dormant splits cleanly, as [Table](#tbl:c32-exclave-methods) classifies each method group: the externally callable selector ABI, the `ANE_Exclave*` methods, is compiled as inert stubs (`mov w0, #0; ret`), while the lower-case infrastructure methods and the proxy service are real.

| Method group | Status on the M1 | Status on the M5 | Examples |
| --- | --- | --- | --- |
| enablement and binding | real bodies | operative | `checkExclaveEnablementStatus`, `setupANEExclaveProxyService`, `ANEExclaveInit` |
| interrupt and worker plumbing | real bodies | operative | `aneExclaveInterruptHandler`, `aneExclaveUpcallEventHandler`, `aneExclaveWorkerThreadEntry` |
| query, handoff, and recovery | real bodies | operative | `aneExclaveQuery`, `aneSEPToExclaveHandoff`, `aneExclaveStartFWRecoveryProcess` |
| operational selector ABI | inert stubs | operative | `ANE_ExclaveLoad`, `ANE_ExclaveEvaluate`, `ANE_ExclaveModeCycle`, `ANE_ExclaveSaveState` |

Table: The exclave method set, classified by status on the M1 and the M5: real implementations and inert stubs on the M1, all operative on the M5. {#tbl:c32-exclave-methods}

The host-side gate is a two-tier check.
A platform call reports whether exclaves are available, and a driver flag reports whether the feature is enabled.
The driver-internal proxy service `ANEExclaveProxy` is a fully implemented IOService subclass with real lifecycle bodies, but it binds only when an exclave device-tree node exists, and that node is declared only by a personality the later-generation kext has and the M1 kext lacks.
On the M1 the platform call reports unavailable: there is no exclave device-tree node, no exclave proxy binds, and no matching exclave core is present in the firmware bundle, so the proxy is never set up and the live behavior cannot be exercised.

On the M5 this dormant path is the live execution substrate, which confirms the later-generation prediction above.
The exclave device-tree node is present, the proxy binds, and the model program loads and runs inside a capability-scoped Swift secure component, `com.apple.aneexclave`, reached from the kernel only through a typed Tightbeam channel.
The component holds explicit segment-access capabilities and nothing more: the seven per-client address-translation contexts the IORegistry exposes as `mapper-ane0-iso1` through `iso7`, plus thirty-two exclave memory regions.
Because the program executes behind this boundary, the per-task-descriptor performance counters are produced secure-side and withheld from an unentitled host, which is why their live values stay gated.
The M5 ran with System Integrity Protection enabled, so this is the enforced posture rather than an artifact of lowered security, and it adds a structural capability and address-translation layer to the temporal single-engine isolation the M1 provides.

## Cross-process isolation and the timing oracle

The engine is single-in-flight per die and shares firmware and DART state across clients, so two processes that both submit work contend for one queue.
That contention produces a cross-process timing side-channel, confirmed on the M1, that leaks engine occupancy while leaving data confidentiality intact.

A measuring process running a single tiny matmul in a continuous zero-copy loop is near the dispatch floor, where it is sensitive to anything that delays its turn at the engine.
Alone, that process reads a median per-call latency of 153 microseconds.
The moment a second process holds the engine with a heavy compute-bound load, the measuring process jumps to a median of about 355 microseconds, a 2.3x increase, because the shared single-in-flight queue serializes its calls behind the contender's.
Toggling the contender on and off produces a distinct square wave in the victim's latency that tracks the schedule at the exact toggle edges with no false transitions across a 10-second run.
[Table](#tbl:c32-victim-latency) gives the victim per-call latency as the contender toggles between idle and heavy.

| Contender state | Victim latency p50 | samples |
| --- | --- | --- |
| off (idle) | 152 microseconds | 32 499 |
| on (heavy) | 355 microseconds | 10 943 |
| delta | +202 microseconds (2.33x) | |

Table: Victim per-call latency tracking a contender toggled on and off on a shared M1 engine, bucketed at 250 ms. {#tbl:c32-victim-latency}

The channel reads occupancy, not workload magnitude.
A near-fixed serialization step of about +178 microseconds appears the instant the contender holds the queue at all, so even its tiniest workload pays almost the full penalty.
Beyond that step there is only a weak monotonic trend, from 333 microseconds to 395 microseconds as the contender's work grows by a factor of roughly 64.
The victim thus reliably learns that a co-tenant is running and a rough busy fraction, but cannot finely size individual operations from latency alone.
The signal is engine-queue serialization rather than unified-memory bandwidth: a concurrent 1 GB-per-loop CPU memory-copy hog moves the victim's latency by only 4 microseconds (a factor of 1.03), so only engine contention drives it.

As a covert or signature channel the resolution is about 20 to 50 ms.
Single contender pulses down to 20 ms were each detected.
A random 16-bit pattern sent at 0.4 s per symbol was recovered at 14 of 16 bits with a median-threshold detector, with the two tail errors attributable to end-of-run clock drift rather than channel capacity.
This puts the practical channel at roughly 20 to 50 bit/s, enough to read a co-tenant's presence, the timing of an inference, and a coarse duty cycle.

Data isolation holds under the same contention.
Three processes ran distinct programs concurrently, each with distinct seeds and weights and each self-checking its output against its own fp32 reference for 3000 rounds.
All 9000 of 9000 checks were correct with zero mismatches and a maximum relative error of 0.0003, which is pure fp16 rounding.
No output ever reflected another process's inputs or weights.
A separate run held 60 distinct program handles across three processes (20 each) and re-executed every one after all were loaded, with 60 of 60 compiling and re-running and no cross-process eviction at that scale.

The root cause is single-in-flight-per-die scheduling, and there is no per-call isolation control exposed; masking a sensitive workload's duty cycle requires coarse-grain time-slicing, batching, or constant-rate padding above the queue rather than any engine setting.
