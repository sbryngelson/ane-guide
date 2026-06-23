# 31. Power and thermal

> The engine runs at one fixed clock with no voltage or frequency scaling, and manages power through a peak-power regulator armed once at power-up by five loop-free stores with the control word `0x11`.
> Idle reads 0 W with rails off, a sustained convolution holds flat at about 1.66 W over 176 seconds, and the densest peak reaches about 4.7 W at 6.96 TFLOP/s.
> The firmware has no thermal logic and reads no temperature; thermal protection is off-engine in the system power manager.

The engine runs at one fixed operating point and manages power through a peak-power regulator, not a frequency sequence.
The system power manager handles voltage and temperature off-engine.

## Boot and device tree

The engine appears in the device tree as a single node `ane0@84000000`, class `AppleARMIODevice`, `compatible = "ane,t8020"`, `device_type = "ane"`.
The match key `ane,t8020` is the kext bind key even on a later die, since the engine block has the t8020-generation identity string across the whole M1 family.
[Table](#tbl:c31-devicetree) gives the properties of that `ane0` node, with the decoded value and meaning of each.

| Property | Decoded value | Meaning |
| --- | --- | --- |
| `compatible` | `"ane,t8020"` | the kext match key |
| `ane-type` | `0x60` | engine variant identifier |
| `reg` bank 0 | `0x2_8400_0000`, length `0x200_0000` | the 32 MB engine control aperture |
| `reg` bank 1 | `0x2_8E08_0000`, length `0xC02C` | the power-manager slice |
| `interrupts` | `0x302` | the engine interrupt number |
| `clock-ids` | `0x13e`, `0x13f`, `0x140`, `0x141` | four clock identifiers, one per compute cluster |
| `clock-gates`, `power-gates` | `0x1cf` | the single clock and power-gate index for the whole block |
| `asc-dram-mask` | `0x1F0_0000_0000` | the high-address window the coprocessor may reach in DRAM |
| `segment-names` | `__TEXT;__DATA` | the firmware segments the loader places |
| `pre-loaded` | `1` | the firmware is resident in DRAM before the kernel attaches |

Table: The properties of the `ane0` device-tree node, with decoded value and meaning. {#tbl:c31-devicetree}

Bank 0 is the engine control aperture at absolute physical `0x2_8400_0000`, 32 MB, holding the coprocessor registers, mailbox doorbells, and engine register file.
Bank 1 is a small power-manager slice at `0x2_8E08_0000`, length `0xC02C`, whose base matches the base of the system power-manager node `pmgr@8E080000`, tying the engine's power-gate `0x1cf` directly to the power-management register neighborhood.
The four clock identifiers `0x13e` through `0x141` are one per compute cluster, matching the four-cluster geometry of the power model.

The firmware is not read from disk at first boot.
The boot loader verifies the signed firmware image and pre-loads its segments into the carve-out named by the node's `segment-ranges`, marked by the property `pre-loaded = 1`.
The kernel driver then attaches to `ane0`, hands the already-trusted image to the engine's coprocessor over the mailbox, and the coprocessor boots.
The image itself is a bare uncompressed and unencrypted preload Mach-O wrapped in an Image4 container with the payload tag `anef`, and it has no embedded manifest, keybag, or code-signature load command of its own.
The per-device personalized boot ticket enforces its authenticity externally, listing the `anef` object digest alongside the kernel, boot-loader, and device-tree digests and binding to the device by its unique chip identifier.
A firmware image cannot be transplanted to a different machine.

The power and peak-power tunables the firmware consumes are not on `ane0`.
They are on `pmgr@8E080000`: the master enable `ane-dpe = 1`, calibration vector `dpe-ane-data` (ten 32-bit words), and per-operating-point current ceiling table `ifane-max`.
The ten calibration words decode to `[9384, 661, 1323, 2641, 2395, 84489, 163813, 51491, 178500, 51491]`, the per-platform coefficients that convert dynamic-power-estimation activity counts to energy, with the repeated value `51491` as a shared scale factor.
The `ifane-max` table is rows of an index key and three current ceilings: the head rows are index `0x1f8` with `{0x18000, 0x17333, 0x17333}`, index `0x2e8` with `{0x1fd70, 0x1fd70, 0x1fd70}`, index `0x498` with `{0x2deb8, 0x2deb8, 0x2deb8}`, index `0x6a8` with `{0x428f5, 0x428f5, 0x428f5}`, and index `0x858` with `{0x628c5, 0x65666, 0x65666}`.
The first column is a monotone activity-keyed operating point, and the three trailing columns are the peak-current ceilings, one per energy-accumulator partition, that the peak-power regulator clamps against.
A parameter-push call at init delivers these, and the firmware arms the power blocks from them.

## Power model

[Table](#tbl:c31-power-facts) collects the decoded boot, power, and clocking facts with their values and sources.

| Fact | Value | Source |
| --- | --- | --- |
| Engine node | `ane0@84000000`, `compatible "ane,t8020"` | device tree |
| Engine aperture | `0x2_8400_0000`, 32 MB | device tree |
| Power-manager slice | `0x2_8E08_0000`, length `0xC02C` | device tree |
| Power-block base | `0x2_6b8f_0000` | firmware disassembly |
| Voltage base (opaque) | `0x2_3b70_c008` | firmware disassembly |
| Power-block arm | 5 fixed stores, no loop | firmware disassembly |
| Peak-power control word | `0x11` | firmware disassembly |
| Compute clusters | 4, gated independently | device tree and firmware |
| Idle power | 0 W, rails off | M1/H13 measured |
| Sustained convolution | $\approx 1.66$ W flat over 176 s | M1/H13 measured |
| Densest peak | $\approx 4.7$ W, $6.96$ TFLOP/s | M1/H13 measured |

Table: The decoded boot, power, and clocking facts; the register-init constants are decoded in Appendix C. {#tbl:c31-power-facts}

The engine runs at a single fixed clock with no local voltage or frequency scaling of its own; any frequency change is the system power manager's to make externally.
The only clock string in the firmware is the boot-time timebase report; the image holds no frequency table, no enumerated operating points, and no voltage string.
Voltage is the system power manager's concern, reached only through an opaque power-management base address (`0x2_3b70_c008`) that the firmware never interprets.

Two controls modulate performance, neither of them frequency.
The first is how many of the four compute clusters are powered.
The second is a peak-power regulator that limits activity under a fixed budget.
That regulator is the engine's substitute for a frequency and voltage sequence, and it has three parts the firmware arms once at power-up.
Dynamic power estimation counts switching activity to estimate instantaneous dynamic power.
Peak-power tracking watches that estimate against the budget and applies back-pressure on issue when the work would exceed it, throttling activity within the fixed clock rather than dropping frequency.
Leakage and energy estimation wires the result into the system power manager.
Three accumulator partitions hold the running energy total that the energy-model telemetry channel reports.

Arming this model is loop-free and idempotent, the five memory-mapped stores of [Listing](#lst:c31-arm-seq) at the fixed base `0x2_6b8f_0000`, gated on two per-instance enable flags at offsets `+0x95` and `+0x96`, with no ramp and no per-frequency programming.

```text
this+0x95 (byte)  : dynamic-power-estimation mode enabled?   (tested at entry)
this+0x96 (byte)  : peak-power-tracking mode enabled?

store  0x11             -> [0x2_6b8f_0000]   ; peak-power-tracking control word
store  (this+0x96 ? 1 : 0) -> [0x2_6b8f_0004] ; dynamic-power-estimation control word
reg = [0x2_6b8f_4000]; reg |= 1; store -> [0x2_6b8f_4000]  ; leakage-and-energy enable bit
store  0x3fff           -> [0x2_6b8e_c42c]   ; trailing control word
store  0xf              -> [0x2_6b8e_c5dc]   ; trailing control word (= 0x2_6b8e_c42c + 0x1b0)
```

Listing: The five-store arm sequence for the power-estimation and peak-power blocks, with the fixed control words at their decoded memory-mapped offsets. {#lst:c31-arm-seq}

The peak-power control word is `0x11`, the leakage-and-energy block is enabled by setting bit 0 of the register at `0x2_6b8f_4000`, and the sequence finishes with two fixed trailing control words, `0x3fff` at `0x2_6b8e_c42c` and `0xf` at `0x2_6b8e_c5dc`.
This estimation aperture at `0x2_6b8f_0000` is distinct from the power-gating base `0x2_3b70_c008`, so power gating and power estimation are separate hardware blocks.
The system-on-chip leakage-and-energy enable bit folds the result into the system power manager.
The rails drop fully when the engine goes idle, so the estimation block loses state and the firmware re-applies the retained calibration on every wake.

The four compute clusters gate independently through the stride-eight status array of [Table](#tbl:c31-power-domains), polled by the host driver until each domain's low byte reads `0xff`, meaning all eight power straps have settled.

| Aperture-relative offset | Poll | Domain |
| --- | --- | --- |
| `0x2c8` | `expect 0xff, mask 0xff, retries 5000` | top-level ready and fabric gate, validated first |
| `0xc000` | `expect 0xff, mask 0xff, retries 5000` | base domain, the always-on control fabric |
| `0xc008` | `expect 0xff, mask 0xff, retries 5000` | compute cluster 1 |
| `0xc010` | `expect 0xff, mask 0xff, retries 5000` | compute cluster 2 |
| `0xc018` | `expect 0xff, mask 0xff, retries 5000` | compute cluster 3 |
| `0xc020` | `expect 0xff, mask 0xff, retries 5000` | compute cluster 4 |

Table: The power-domain status registers, a stride-eight array at `0xc000 + domain*8` polled until each low byte reads `0xff`. {#tbl:c31-power-domains}

The five domains are one always-on base domain and four independently gated compute clusters; there is no fifth cluster.
The firmware brings the base domain up first and adds a compute cluster only when work needs it, which is the mechanism behind the measured idle of 0 W.
The engine is in the all-off state until a procedure call arrives, then brings the base domain up to a wait state and gates in the clusters, so the idle floor is rail-off rather than clock-gated.
A free-running firmware timestamp pair is in the same aperture at `0x1170000` for the low word and `0x1170004` for the high word, distinct from the engine-internal timebase, and the firmware zeroes four scratch registers at `0x1840048` through `0x1840054` at init.

For a workload held at the operating point for time $t$, the energy is

$$E = P \, t$$

with measured constants from a sustained saturating loop on the M1: a dense convolution at $P \approx 1.66 \text{ W}$ held flat over $t = 176 \text{ s}$, and a densest-packed peak of $P \approx 4.7 \text{ W}$.
At that peak the engine delivers $6.96$ TFLOP/s, which gives an efficiency of about $1.5$ TFLOP/s per watt in fp16, rising to about $2.6$ TFLOP/s per watt on weight-reuse convolutions.
The power scales with cluster engagement and lane density, not with any clock change.
Relative to a single cluster, throughput rises $1.99\times$, $3.02\times$, and $4.00\times$ at the second, third, and fourth clusters.
The rail then climbs smoothly from $779$ mW to $1429$ mW as channels pack the four clusters, with no discrete steps.

## Thermal behavior and the operating point

A scan of the image for temperature, throttle, junction, and similar terms returns nothing.
The engine does not read temperature, does not throttle on temperature, and emits no thermal event of its own.
Thermal protection is entirely off-engine: the peak-power regulator bounds power, which indirectly bounds heat, and package-level thermal control is in the system power manager, which can delay or refuse dispatch but does not reach into any engine counter.

A 176-second saturating loop sampled thermal pressure every two seconds and read nominal on every one of the 89 samples, with power and throughput flat to within half a percent and a slightly negative drift.
A frequency-scaling engine would show a ramp or multiple power modes as it settled; a thermally limited engine would decay.
The engine does neither; power and throughput hold flat.
The single outbound power signal the firmware emits is a normalized margin level for the host to react to, holding no thermal field.

The first dispatch after idle pays a fixed power-up cost because the engine reaches a true rails-off state between jobs.
That cost grows from near zero at back-to-back dispatch to about $0.5$ ms once the idle gap reaches roughly 100 ms, then plateaus, which sets the residency window for keeping the engine warm.

### Operating points and the absence of local DVFS

The engine has no frequency-or-voltage sequence of its own.
The kext delegates operating-point selection to the system power manager: the SoC CLPC against ApplePMGR.
The kext holds no clock register addresses.
A disassembly of `enableAneSysClock` shows it performs no memory-mapped writes itself; it loads an ApplePMGR service object, authenticates the vtable pointer, and calls a PMGR method with the enable argument set, with the only literal in the path being the string `"ApplePMGR"`.
The `ane0` device-tree node publishes the PMGR clock and power-gate identifiers but has no `dvfm-states`, `voltage-states`, or `perf-states` property, so the per-state frequency in Hz and voltage in mV stay held privately by ApplePMGR behind the power-manager base and are not recoverable from the engine binaries.

What is on the engine side is the firmware `H13TunableManager` register-init table, where `H13` is the M1 family.
A `TunableManager` descriptor at base `0x2_6b8f_4000` (`aneDpePpt_soc_dpe_lee`, the SoC dynamic-power-estimation control block) holds the 7-step monotonic credit sequence of [Table](#tbl:c31-tunable), written once at firmware bring-up and gated by chip revision.

| Offset | Mask | Value | Decoded |
| --- | --- | --- | ---: |
| `+0x18` | `0x1ff` | `0x19` | 25 |
| `+0x1c` | `0x1ff` | `0x32` | 50 |
| `+0x20` | `0x1ff` | `0x46` | 70 |
| `+0x24` | `0x1ff` | `0x55` | 85 |
| `+0x28` | `0x1ff` | `0x5f` | 95 |
| `+0x2c` | `0x1ff` | `0x69` | 105 |
| `+0x30` | `0x1ff` | `0x73` | 115 |

Table: The 7-step monotonic DPE/PPT credit sequence in the `aneDpePpt_soc_dpe_lee` block, with the decoded value of each 9-bit field. {#tbl:c31-tunable}

These nine-bit fields are the per-operating-point power-credit and peak-power-throttle thresholds the on-die estimator caps activity against, indexed by the perf state the CLPC selects.
Their strictly increasing sequence is the signature of a seven-state perf sequence.
The engine has seven operating points whose registers hold throttle credits, not a clock frequency and not a voltage.
A parallel eight-step sequence for the ASC block holds the values `4, 7, 13, 20, 27, 36, 46, 59`.

Live read-only dtrace confirms the engine holds one power state through sustained work.
A six-layer matmul-and-relu program dispatched in a continuous loop for 16 seconds drove 56,527 dispatches, traced entry-only with no destructive flag.
`ChangePowerState`, `populateCLPCPerfInfo`, `power_on_hardware`, and `setPowerStateGated` each fired exactly once at warm-up and never again across the 56,527 dispatches.
Every dispatch instead walked a fixed path: `EnableANEClocksAndPower` to un-gate, one `submitWorkToPerfController` that hands an `ANEPerfRequest` and a modeled-performance hint to the CLPC, a `notifyPerfController` start-and-end pair, then `enableDynPowerGating_gated` to re-arm the idle gate.
The kext does not walk a frequency sequence per submit; the engine holds a single power-domain state for the whole run while the CLPC drives any frequency change internally and invisibly to the kext.
This is the runtime counterpart to the flat power and throughput measured under sustained load: the firmware tracks a single PerfMode, and setting it twice is a no-op.
