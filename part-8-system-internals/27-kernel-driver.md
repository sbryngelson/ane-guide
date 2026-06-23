# 27. Kernel driver and IOKit ABI

> The engine is reached through a single kernel driver and a flat user-client selector ABI, with 17 control-client selectors and 9 direct-path selectors, each pinned to one exact size tuple.
> Opening either user client is gated on the kernel entitlement `com.apple.ane.iokit-user-access`, which exactly two system binaries hold, so every application reaches the engine through a privileged broker daemon.
> Selector 2 alone reaches a hardware doorbell from user space, through a four-layer call path ending in an MMIO mailbox write.
> The size tuples match across the M1 and M2-class kernel cache, so the ABI at this layer is family-invariant.

This chapter covers the driver class hierarchy, the two user-client selector spaces with their exact struct sizes, the dispatch-array record format the kernel validates against, the path from a user-space call down to the hardware doorbell, and the broker model that fronts the device with one privileged daemon.

## Driver stack and class hierarchy

The three cooperating kexts, their bundles, principal classes, and provider matches appear in [Table](#tbl:c27-kext-stack).

| Kext role | Bundle | Principal class | Provider match |
| --- | --- | --- | --- |
| interface and hardware driver | `AppleH11ANEInterface` | `ANEHWDevice` (registered as `H11ANEIn`) | `RTBuddyService`, role `ANE` |
| per-die abstraction layer | `AppleT8132ANEHAL` | `AppleT8132ANEHAL` | `IOResources` |
| multi-engine arbiter | `AppleANELoadBalancer` | `ANEDriver` (`H1xANELoadBalancer`) | `IOResources`, `IOKit` |

Table: Bundle, principal class, and provider match for each of the three kernel extensions. {#tbl:c27-kext-stack}

Three version-locked kexts cooperate, all at build `9.511.3` on the M1 generation.
The interface and hardware driver registers the device, vends the user clients, and rings the firmware.
A per-die hardware abstraction layer supplies the clock, power, and topology constants.
The load balancer owns the program-to-engine residency map and arbitrates across physical engines on parts that have more than one.
The engine attaches as an Apple RTKit coprocessor endpoint, so the interface driver is the host side of a real-time-operating-system mailbox client.
Three kernel families handle the work: the address-translation family that drives the device IOMMU, surface family that backs zero-copy tensor buffers, and real-time mailbox family that holds firmware commands.

`ANEHWDevice::newUserClient(task*, void*, uint type, IOUserClient**)` vends the two distinct user clients plus a hint-only client of [Listing](#lst:c27-userclients).

```c
/* ANEHWDevice::newUserClient vends, by requested type: */
H11ANEInUserClient        /* control client: program lifecycle + the inference hot path */
H11ANEInDirectPathClient  /* direct path: enqueue, memory-map, session-hint            */
ANEClientHints            /* scheduling-hint client (setClientHint)                    */
```

Listing: The three user-client types, by requested type. {#lst:c27-userclients}

The two functional clients hold separate connections and separate selector spaces that both start at zero, so one small selector integer names different methods on each.
Each stores its connection at object offset `+0x40`, and every selector call loads that offset before issuing the kernel call.

On the user-space side an `IOServiceOpen` whose return type is `0xe00002c5` opens the device, and the connection object is at offset `0x40` in the device handle.
Two front-ends open that type: `ANEServicesDevice` for inference and `ANEHWDevice` for administration.
The user-space selector immediates read on the M1 are selector 0 for the device open, selector 2 for the send-request, selector 3 for create, and selector 4 for prepare.
The rest are selector 6 for destroy, selector 10 for the version query, and selector 16 for the firmware load.
The lifecycle indices differ from the kernel-registered table because the kernel has two user-client classes with independent selector tables.
The two selectors agree across both sides, selector 0 opening the device at 104 bytes and selector 2 sending a request with a 2376-byte input and a 40-byte output.

Every selector shim reads its arguments out of the framework `IOExternalMethodArguments` block at the fixed set of offsets in [Listing](#lst:c27-extmethodargs), identical across all 26 selectors.

```c
/* IOExternalMethodArguments field offsets, used by every selector shim: */
+0x08  asyncWakePort       /* mach completion port (async selectors only)        */
+0x10  scalarInput[]       /* also holds the 0x20-byte asyncReference block       */
+0x20  scalarInputCount
+0x30  structureInput      /* pointer to the typed argument struct                */
+0x38  structureInputSize
+0x48  scalarOutput[]
+0x50  scalarOutputCount
+0x58  structureOutput
+0x60  structureOutputSize
```

Listing: The `IOExternalMethodArguments` field offsets every selector shim reads. {#lst:c27-extmethodargs}

Three return constants recur across the shims: `0xe00002c2` is `kIOReturnBadArgument`, returned on a size or null-pointer check failure, `0xe00002c7` is `kIOReturnUnsupported`, returned on a disabled or stub path, and `0xe00002c5` is the closed-or-closing client state.

## User-client dispatch-array format

Each user client routes a selector through the standard IOKit 2022 dispatch pattern.
`H11ANEInUserClient::externalMethod` loads the dispatch-array pointer and the method count, then tail-calls the framework dispatcher, which validates the declared scalar and structure sizes against the array entry before it calls the named handler.
Both dispatch arrays were read byte for byte out of the read-only data section of the kernel cache, and [Listing](#lst:c27-dispatch-disasm) gives the disassembled selector dispatch for each user client with its array pointer and selector count.

```c
/* H11ANEInUserClient::externalMethod, disassembled (M1, T6000): */
add x3, x3, #0xc08    /* x3 = &_sANEDriverClientMethods           */
mov w4, #0x11         /* count = 17 selectors (0..16)             */
bl  IOUserClient2022::dispatchExternalMethod

/* H11ANEInDirectPathClient::externalMethod: */
add x3, x3, #0xeb0    /* x3 = &_sANEDriverDirectPathClientMethods */
mov w4, #0x9          /* count = 9 selectors (0..8)               */
bl  IOUserClient2022::dispatchExternalMethod
```

Listing: The disassembled selector dispatch for each user client, with its dispatch-array pointer and selector count. {#lst:c27-dispatch-disasm}

Each array element is one `IOExternalMethodDispatch2022` record at a 40-byte stride, packing the authenticated handler pointer and the four size checks the dispatcher enforces, shown in [Listing](#lst:c27-dispatch-elem).

```c
/* IOExternalMethodDispatch2022 element, 40-byte stride, field offsets: */
struct IOExternalMethodDispatch2022 {
  void *function;                  /* +0x00  pointer-authenticated handler   */
  uint32_t checkScalarInputCount;  /* +0x08  exact scalar-input count        */
  uint32_t checkStructureInputSize;/* +0x0c  exact struct-input bytes        */
  uint32_t checkScalarOutputCount; /* +0x10  exact scalar-output count       */
  uint32_t checkStructureOutputSize;/* +0x14 exact struct-output bytes       */
  uint8_t  reserved[0x10];         /* +0x18  reserved (debug-WP group flag)  */
};
```

Listing: The dispatch-array element layout, packing the handler pointer and the four size checks the dispatcher enforces. {#lst:c27-dispatch-elem}

No entry on either client uses the sentinel `0xffffffff` that means "do not check".
Every selector pins an exact scalar count and an exact struct size, and the dispatcher rejects any other size with `kIOReturnBadArgument`.
Size-based overloading does not exist here: each selector index has exactly one record with one fixed size tuple.

## Control-client selector table

The control client has 17 selectors covering the open handshake, the program lifecycle, status and version reads, and the firmware-driven debug work-processor channel.
[Table](#tbl:c27-control-selectors) lists every control-client selector with its handler and kernel-authoritative sizes read from the dispatch array.

| Sel | Handler | Scalar in | Struct in | Scalar out | Struct out |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `ANE_DeviceOpen` | 0 | 104 | 0 | 104 |
| 1 | `ANE_DeviceClose` | 0 | 0 | 0 | 0 |
| 2 | `ANE_ProgramSendRequest` | 1 | 2376 | 0 | 40 |
| 3 | `ANE_ProgramCreate` | 0 | 32 | 0 | 0 |
| 4 | `ANE_ProgramPrepare` | 0 | 56 | 0 | 56 |
| 5 | `ANE_ProgramUnprepare` | 0 | 56 | 0 | 0 |
| 6 | `ANE_ProgramDestroy` | 0 | 16 | 0 | 0 |
| 7 | `ANE_GetStatus` | 0 | 0 | 0 | 32 |
| 8 | `ANE_ProgramCreateInstance` | 0 | 32 | 0 | 0 |
| 9 | `ANE_ProgramChainingPrepare` | 0 | 16 | 0 | 24 |
| 10 | `ANE_GetVersion` | 0 | 0 | 1 | 0 |
| 11 | `ANE_RegisterDebugWorkProcessor` | 0 | 24 | 0 | 0 |
| 12 | `ANE_UnregisterDebugWorkProcessor` | 0 | 0 | 0 | 0 |
| 13 | `ANE_GetDebugWorkProcessorItem` | 2 | 0 | 0 | 0 |
| 14 | `ANE_CompleteDebugWorkProcessorItem` | 2 | 0 | 0 | 0 |
| 15 | `ANE_ReleaseDebugWorkProcessorBuffers` | 0 | 0 | 0 | 0 |
| 16 | `ANE_LoadFirmware` | 3 | 0 | 0 | 0 |

Table: The seventeen control-client selectors, with handler name and kernel-authoritative scalar and structure sizes. {#tbl:c27-control-selectors}

Selector 0 is the open handshake.
It passes a 104-byte device-info structure as both the input and the output buffer, echoes the caller header back, fills the output half with the device descriptor, and returns the session token at offset `+0x00` that every later request holds, with the handshake fields given in [Table](#tbl:c27-deviceinfo).

| Offset | Input (client to kernel) | Output (kernel to client) |
| --- | --- | --- |
| `+0x00` | usage type byte (1 standard, 2 unsupported) | program / session token (u64) |
| `+0x08` | callback function pointer; board id `0x1111222233334444` | echoed |
| `+0x10` | receiver context pointer | echoed |
| `+0x18` | timeout `0x2710` = 10000 | echoed |
| `+0x48` | (output only) | ANE version `0x20` = 32, 256 |
| `+0x50` | (output only) | number of engines = 1 |
| `+0x60` | (output only) | CPU subtype = 4 |

Table: The `ANEDeviceInfo` handshake structure passed in and echoed back by selector 0. {#tbl:c27-deviceinfo}

The usage-type byte selects the standard client profile: usage `1` opens, usage `2` returns the unsupported code `24`.
Selector 16 is inactive on this build: its shim returns `kIOReturnUnsupported` unconditionally, and the real firmware load runs internally at driver start.

A compiled program reaches the kernel in one of two representations: `ANEProgramLegacyResource`, a loader for the program-image executable, and `ANEProgramRTResource`, a runtime op-graph variant.

## Direct-path selector table

The direct-path client has 9 selectors, listed in [Table](#tbl:c27-directpath-selectors) with their handlers and sizes.
Selectors 0, 1, and 2 reuse the control client's handler functions; the remaining six are the enqueue, memory-map, and session-hint methods of the low-latency submission model.

| Sel | Handler | Scalar in | Struct in | Scalar out | Struct out |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `ANE_DeviceOpen` | 0 | 104 | 0 | 104 |
| 1 | `ANE_DeviceClose` | 0 | 0 | 0 | 0 |
| 2 | `ANE_ProgramSendRequest` | 1 | 2376 | 0 | 40 |
| 3 | `ANE_ProgramOutputSetEnqueue` | 0 | 40 | 0 | 0 |
| 4 | `ANE_ProgramInputsReady` | 0 | 3104 | 0 | 0 |
| 5 | `ANE_MemoryMapRequest` | 1 | 2080 | 1 | 0 |
| 6 | `ANE_MemoryUnMapRequest` | 0 | 2080 | 0 | 0 |
| 7 | `ANE_SessionHintRequest` | 0 | 16 | 0 | 24 |
| 8 | `ANE_ProgramChainingSetActiveProcedure` | 0 | 32 | 0 | 0 |

Table: The nine direct-path selectors with their handler names and scalar and structure sizes; the full reference is in Appendix C. {#tbl:c27-directpath-selectors}

Selector 5 is the device-IOMMU map.
Its 2080-byte parameter structure describes a host buffer, and on success the handler writes the resulting engine-visible device address back into the single scalar output slot.
Selectors 3 and 4 are the pre-post and trigger of the resident submission model: an output buffer set is enqueued, the inputs-ready signal fires, and the same doorbell path as selector 2 rings the engine.

## Register and exclave method catalog

Beyond the nine kernel selectors, the direct-path client exports a wider register, power, firmware, and secure-world method surface.
These are not distinct kernel selector indices: each routes through one of the nine kernel selectors or through a separate entry point, and [Table](#tbl:c27-method-catalog) names the surface by role.

| Method | Role |
| --- | --- |
| `ANE_PowerOn` / `ANE_PowerOff` / `ANE_IsPowered` | power-domain control |
| `ANE_LoadFirmware` / `ANE_ForgetFirmware` | firmware image lifecycle |
| `ANE_SendCommand` | raw firmware command injection |
| `ANE_SetPowerManagement` / `ANE_SetDynamicPowerGating` / `ANE_SetPowerGatingHysteresisTime` | power policy |
| `ANE_SetThrottlingPercentage` | thermal throttle |
| `ANE_SetDARTCacheTTL` / `ANE_FlushInactiveDARTMappings` / `ANE_UnmapDartBuffers` | address-translation controls |
| `ANE_ReadANERegister` / `ANE_WriteANERegister` | raw memory-mapped register read and write |
| `ANE_FWSharedEventDoorbellRing` | ring the firmware shared-event doorbell |
| `ANE_AddPersistentClient` / `ANE_RemovePersistentClient` | keep the device resident |
| `ANE_MPMMemoryMapRequest` / `ANE_MPMMemoryUnmapRequest` | the multi-process managed-memory region |
| `ANE_ExclaveCycle` / `ANE_ExclaveLoad` / `ANE_ExclaveEvaluate` / `ANE_ExclaveUnload` | secure-world load and evaluate |
| `ANE_ExclaveReadPropertyValue` / `ANE_ExclaveWritePropertyValue` | secure-world property access |
| `ANE_GetClientsInfo` / `ANE_ShowSharedMemoryAllocations` / `ANE_ShowModelMemoryStatus` | diagnostics |

Table: The register, power, firmware, and exclave method surface exported by the direct-path client beyond its nine kernel selectors. {#tbl:c27-method-catalog}

A second access check beyond the kernel entitlement gates the raw register read and write, command injection, and exclave methods: a privileged-virtual-machine-access property probed at client open, distinct from the device-open entitlement.

## From a user-space call to the doorbell

A submit on selector 2 crosses four layers from the user-space call down to the hardware doorbell write, traced in [Listing](#lst:c27-doorbell-path).

```c
/* The submit path for selector 2 / direct-path selector 4: */
H11ANEInUserClient::externalMethod(sel=2, args)
  -> dispatchExternalMethod                 /* validates 2376-in / 40-out */
  -> ANE_ProgramSendRequest(client, ref, args)            /* arg shim     */
  -> ANEClientDevice::programSendRequest(ANEProgramRequestArgs*, ...)
  -> ANEDriver::ANE_ProgramSendRequest(...)               /* gated        */
  -> ANEHWDevice::doorBellRing(db)
  -> ANERegisterControl::write32(reg, 1 << idx)           /* MMIO mailbox */
```

Listing: The four-layer call path from a user-space submit selector down to the hardware doorbell write. {#lst:c27-doorbell-path}

Below the dispatcher, the thin shim re-checks the argument sizes and unmarshals the typed argument structure, the client object method builds the memory descriptors and retains the shared-event fences, and the gated driver method runs on the command-gate workloop.
The 2376-byte request structure holds the program handle minted at create time, a sequence number, the quality-of-service and execution-priority pair, and the array of surface identifiers for the input, output, and intermediate buffers, with the measured field layout in [Table](#tbl:c27-request-struct).

| Offset | Field | Observed |
| --- | --- | --- |
| `+0x000` | program / instance token (u64) | the handle minted by program-create |
| `+0x008` | sequence number | 0, then 1 on the next submit |
| `+0x010` | priority / quality-of-service pair | `(5, 21)`, qos class and execution priority |
| `+0x01c` | io category | 2 |
| `+0x020` | surface identifier array | input, output, intermediate surfaces |

Table: The measured field layout of the 2376-byte request structure submitted on selector 2. {#tbl:c27-request-struct}

The 40-byte output returns the sequence and the echoed token: `+0x00` is the sequence and result, `+0x08` is the echoed token, and `+0x20` is a status flag.
Selector 2 alone uses the asynchronous machinery, and it is the only path that reaches a hardware doorbell from user space.
Completion arrives at a mach wake port as a callback, not by a shared-memory poll.
The doorbell write itself reads the doorbell index from the request, requires it below 32, computes the mask `1 << index`, and stores that mask into the engine register aperture, the mailbox signal that triggers the firmware.

The reverse signal, the engine telling the host a job is complete, travels the same windowed store mechanism in firmware.
The engine rings a host interrupt with an interrupt-atomic memory-mapped store to the host-supplied target register, bracketed by clearing and then setting bit 39 of the implementation-defined AArch64 system register `S3_3_C15_C8_0`.
Clearing bit 39 opens the posted-write window, the firmware stores the doorbell value into the host aperture, and a barrier separates the store from the status sample.
The firmware then reads back the uncorrectable-cache-error bit (bit 1) and the transaction-reject bit (bit 7) to confirm the store has committed before setting bit 39 to close the window.
The whole sequence runs with interrupts disabled so a nested handler cannot corrupt the status read.

## Entitlement gate and broker model

The client open checks the two kernel entitlements of [Listing](#lst:c27-entitlements), the hard device-open gate and the resident data-chaining gate.

```c
/* checked at H11ANEInUserClient::init via copyClientEntitlement: */
"com.apple.ane.iokit-user-access"            /* the hard device-open gate   */
"com.apple.ane.allow-dataChaining-access"    /* resident data-chaining gate */
```

Listing: The two kernel entitlements checked when a user client is opened. {#lst:c27-entitlements}

A single kernel entitlement gates opening either user client.
The check runs once at client construction and is a boolean on the client object, not re-checked per selector.
Across the whole system, exactly two binaries hold `com.apple.ane.iokit-user-access`: the system broker daemon and its per-user sibling.
No application process opens the device.
Every other consumer reaches the engine through the broker over a cross-process call, proving itself with an entitlement from the broker's own private family rather than the kernel gate.

The driver stamps the capability at client creation in `ANEClientInfo::create`, which reads each entitlement through `copyClientEntitlement` and records `isPrivileged` and `allowDataChaining` as bits on the client.
Beyond the two open gates, the driver enforces six further `com.apple.ane` and `com.apple.private.ane` entitlements covering scheduling priority, memory and data access, and client and coalition hints, as [Table](#tbl:c27-driver-entitlements) lists.

| Entitlement | Capability |
| --- | --- |
| `com.apple.ane.realtime-priority-client` | the real-time-priority client grant |
| `com.apple.ane.allow-system-reserved-priorities` | use of the system-reserved scheduling priorities |
| `com.apple.ane.memory` | a memory-access grant |
| `com.apple.ane.allow-data` | a data-access grant |
| `com.apple.private.ane.allow-set-client-hints` | set per-client hints |
| `com.apple.private.ane.allow-share-coalition-hints` | share hints across a coalition |

Table: The kernel-driver entitlements beyond the two open gates, with the capability each grants. {#tbl:c27-driver-entitlements}

The broker keys are a private entitlement family, checked per connection and per method over a cross-process call, with each key, its capability, and its holder count given in [Table](#tbl:c27-entitlement-family).

| Entitlement | Capability | Holders |
| --- | --- | --- |
| `com.apple.ane.iokit-user-access` | the hard kernel gate: open the user client, privileged device open | 2: the broker and its per-user sibling |
| `com.apple.ane.allow-dataChaining-access` | resident data-chaining on the direct-path client | kernel-checked at client init |
| `com.apple.aned.private.allow` | baseline: compile, load, instantiate through the broker | 18 |
| `com.apple.aned.private.ANEAccess.allow` | inference-client access-grant variant | 14 |
| `com.apple.aned.private.adapterWeight.allow` | stream adapter weights onto a shared resident base model | 5 |
| `com.apple.aned.private.processModelShare.allow` | share one resident model across processes | 4 |
| `com.apple.aned.private.secondaryANECompilerServiceAccess.allow` | the longer-duration compiler service for large models | 1 |
| `com.apple.aned.private.aggressivePowerSaving` | the aggressive-power-saving execution mode | gate helper only |
| `com.apple.aned.private.modelPurgeInAllPartitions` | purge models across all cache partitions | gate helper only |
| `com.apple.security.temporary-exception.iokit-user-client-class` | open the direct-path user client for own submission | 27 |
| `com.apple.security.ts.ane-client` | trust-cache blessed-client slot for latency-critical consumers | 5 |

Table: The entitlement family that gates the engine, from a static scan of the M1 system binaries, independent of boot-security state. {#tbl:c27-entitlement-family}

The broker is a listener that enforces per-connection and per-method entitlement checks.
It sorts admitted clients into a restricted tier, unrestricted tier, and per-user tier, and it threads a quality-of-service argument through every compile, load, and instantiate method.
The restricted tier admits the adapter-weight, model-share, and aggressive-power-saving requests through a per-method admission helper, and the per-user tier serves the per-user broker.

The adapter-weight path is the mechanism behind swappable model weights without recompilation.
A base model is loaded once, and each adapter is a new instance bound to a named base-model identifier holding only its per-adapter weight files, through a create-instance-with-weights method that names the base-model identifier and the weight-file count.
Residency and power are explicit per-instance arguments on the create-instance method.
They are an enable-power-saving flag, more-aggressive variant gated by the restricted tier, opt-out-of-model-memory-unwiring flag that keeps a hot client's weights resident at the cost of footprint, and queue-depth that the broker down-adjusts under contention.
A queue-index function and a program-priority function map the quality-of-service argument to hardware scheduling.
A privileged subset of system daemons also holds a sandbox exception that opens the direct-path user client and drives per-inference submission on its own connection, skipping the per-call round trip through the broker.
That exception is a latency optimization, not a capability grant: the privileged device open still happens in the broker, which hands the client a program handle and an intermediate-buffer handle.

The kernel binds residency to code-signing identity.
A second client may attach to an already-resident program only when its team identifier and code-directory hash match the owner, so a shared resident model or key-value cache cannot leak across tenants.
The kernel resolves the caller's team identifier and code-directory hash, and the attach path tests them against the resident owner before a sibling instance reuses the shared intermediate-buffer handle.
With one physical engine on this generation, cross-client arbitration is time-division multiplexing on a single gated request queue, biased by the per-stream quality of service the clients declare.

## Live device properties

The driver publishes its topology and version constants into the registry, read live on the M1 host and decoded in [Table](#tbl:c27-device-props).

| Property | Value | Meaning |
| --- | --- | --- |
| architecture type string | `h13g` | microarchitecture family, drives per-family codegen |
| version | 96 = `0x60` | major hardware version |
| minor version | 17 | minor revision |
| board type | 96 | board and system-on-chip type id |
| board subtype | 0 | board sub-variant |
| number of cores | 16 | compute cores in this engine |
| number of engines | 1 | distinct engine units, load balancer is a pass-through |
| CPU subtype | 4 | program-ABI gate |
| internal build | No | release build, gates the debug surfaces |

Table: The device properties the driver publishes into the registry on the M1 host. {#tbl:c27-device-props}

The number-of-cores value is a topology count and not the throughput-relevant multiply-array width, so a floating-point rate is taken from the measured cost-model anchor rather than inferred from the core count.
At rest the registry shows the device, load-balancer instance, and standing hints client present, with zero control clients and zero direct-path clients open, confirming the brokered, lazily-opened model.
The driver opens the user clients on demand per active client and tears them down when idle.
