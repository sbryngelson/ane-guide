# 10. Power and efficiency

> The engine draws less power than the GPU on every workload class measured, including the ones where the GPU is faster.
> On the M1 it delivers 2 to 14 times the GPU energy efficiency.
> The advantage holds across the M1, M2, and M5, so it is a property of the fixed-function fp16 datapath, not of one chip.

## Headline

Across the measured workload classes the engine runs at a few watts where the GPU runs at tens of watts.
On the M1 the engine delivers 2 to 14 times the GPU's energy efficiency, measured as throughput per watt of idle-subtracted total-package power.
A stack of sixteen 3x3 convolutions sets the high mark: the engine reaches 2063 GFLOP/s per watt against the GPU's 142, a 14.5 times efficiency advantage, while also running about 2 times faster.
The M5 holds the same shape at 2289 GFLOP/s per watt on the engine against 175 on the GPU, a 13 times advantage.

The efficiency lead survives even where the engine loses on latency.
On a large square matrix multiply with inner dimension 4096 the M1 GPU is about 2 times faster in raw throughput, yet the engine still leads on energy.
It computes that workload at 4.4 watts where the GPU draws 32.5 watts, a 4.0 times efficiency edge.
The engine's draw stays in the single digits of watts across these classes; the GPU climbs into the tens.

## Watt-complete map

The map covers the workload classes a perception or encoder deployment is built from.
Convolution appears as a single 3x3 layer and as a sixteen-deep resnet-style stack.
Matrix multiply is measured at three sizes: a dispatch-bound floor, bandwidth-bound mid size, and compute-bound 4096 square.
Attention is measured at a short vision-transformer sequence and a longer 512 sequence.
The normalization family covers layer norm, rms norm, and group norm.
A set of scientific kernels covers a discrete Fourier transform as a matrix multiply, five-point stencil, and fixed-iteration linear solve.

[Table](#tbl:c10-headline) collects the efficiency standouts named above, with the engine and GPU throughput per watt and their ratio on each chip generation.

| Workload | Engine GFLOP/s/W | GPU GFLOP/s/W | Ratio | Silicon |
| --- | --- | --- | --- | --- |
| Conv-resnet stack (16x 3x3) | 2063 | 142 | 14.5x | M1/H13 |
| Conv-resnet stack (16x 3x3) | 2289 | 175 | 13x | M5/H17s |
| Large square GEMM (inner 4096) | 4.4 W | 32.5 W | 4.0x | M1/H13 |
| Five-point stencil (fused) | n/a | n/a | 49x | M5/H17s |
| Workload-class envelope | n/a | n/a | 2 to 14x | M1/H13 |

Table: Engine versus GPU energy efficiency across workloads and chip generations, in GFLOP/s/W or absolute watts. {#tbl:c10-headline}

The large square GEMM row reports absolute package power rather than throughput per watt, because the GPU is faster on that workload on raw throughput while the engine still draws fewer watts to compute it.
The efficiency lead is largest on the convolution stack and on the fused multi-step stencil, where the engine does sustained compute at low power and the GPU spends proportionally more power to keep its units supplied.
The stencil reaches a 49 times energy advantage on the M5, the widest single number in the map.
On the small normalization reductions the lead narrows, with both devices finishing in well under a millisecond and the per-watt numbers close to a tie.
It narrows again on the large 4096 square matrix multiply, which favors the GPU on speed and on fp16 accuracy, though the engine keeps its lower absolute draw there.
Only at the trivial dispatch-bound region, the 256-inner matrix multiply, does the engine lose the efficiency comparison outright, where the FLOP count is too small to matter and the lowest call overhead is the better choice.

## Two-generation result

The same sixteen-class harness was run on an M1 and on an M5, with package power taken from hardware instrumentation on both.
Both silicons return the same verdict: the engine is the efficiency device on every substantial workload class.
The convolution stack reads 2063 GFLOP/s per watt on the M1 and 2289 on the M5, against GPU figures of 142 and 175, so the efficiency advantage holds at 14.5 times and 13 times across the generation gap.
A third run on the M2, an A14 part, fills the middle and reproduces the same shape, with the convolution stack at 2234.6 GFLOP/s per watt on the engine against 172.8 on the GPU, a 12.9 times advantage.
The engine rail never exceeds about 6 watts where the GPU pulls 13 to 21 watts on the same classes.

The per-class efficiency rows on the M2 trace the same split the M1 and M5 maps draw, and [Table](#tbl:c10-m2) gives the per-class engine and GPU throughput per watt with the total-package energy ratio.

| Workload | Engine GFLOP/s/W | GPU GFLOP/s/W | Energy ratio (GPU over engine) |
| --- | ---: | ---: | ---: |
| GEMM floor (M=64, K=256, N=256) | 11.4 | 16.6 | 0.7x |
| GEMM bandwidth (M=128, K=1024, N=1024) | 298.2 | 129.9 | 2.3x |
| GEMM compute (M=256, K=4096, N=4096) | 773.3 | 213.7 | 3.6x |
| Conv single (C=64, 32x32, k=3) | 25.1 | 31.4 | 0.8x |
| Conv resnet stack (C=256, 32x32, k=3, d=16) | 2234.6 | 172.8 | 12.9x |
| Attention ViT (S=197) | 480.8 | 120.2 | 4.0x |
| Attention long sequence (S=512) | 567.4 | 150.1 | 3.8x |
| Five-point stencil (256x256, 32 steps) | 37.6 | 0.8 | 47.8x |

Table: Per-class engine versus GPU efficiency on the M2, an A14 part, with the total-package energy ratio. {#tbl:c10-m2}

The real-model rows on the M2 are the engine's widest margin: a full twelve-layer ViT-B/16 forward runs at 67.9 millijoules per inference against the GPU's 714.3, a 10.5 times energy advantage, and a ResNet-18 forward at 2.6 against 22.3 millijoules per inference, an 8.5 times advantage.
Both also run from 1.7 to 4 times faster.
On the M1 the engine draws under 2.3 watts at its compute roof and spends roughly 0.5 picojoules per FLOP in its fixed-function datapath.

The advantage is wider on the older M1 because that chip is more bandwidth-limited.
The M1 streams at roughly 10 GB/s effective against the M5's roughly 57 GB/s, and the smaller memory bandwidth makes the GPU spend proportionally more power to lead on raw throughput there.
On the M1 the engine sustains a 256-channel 3x3 convolution at 643 GFLOP/s per watt, drawing 1.78 watts, with the rail reading about zero at idle.
The 1.78 watt figure is the canonical short-run draw for this workload, and a longer 176 second run of the same convolution in chapter 31 settles slightly lower at about 1.66 watts.
The two figures are thus the same workload measured over different durations rather than a disagreement.

## Rail-off idle floor

The engine reads about zero milliwatts at idle, and that figure is a property of the power-state machine, not a measurement artifact.
The firmware holds the engine in a fully gated state, `ANE_POWER_STATE_ALL_OFF`, with every domain off, until a job arrives.
Power-up is lazy and job-triggered: the engine is at `ALL_OFF` and brings up its base domain only on the first call, transitioning through `ANE_POWER_STATE_BASE_PS_ON_WAIT` before it gates in the compute sets.
There is no clock-gated but powered idle floor; the idle state is rail-off.

The M1 engine has five independently gated power domains: a base domain for the control and front-end fabric, brought up first, and four compute sets, one per ANE cluster.
Dynamic power-gating, on by default, collapses the compute sets back toward `ALL_OFF` between jobs, so idle work pays no rail power.
Disabling it keeps the sets powered between jobs, trading idle power for lower per-job latency.
Thermal management and voltage are off-engine: the firmware image has no thermal, temperature, or throttle path, so those decisions are in the system-on-chip power manager, not in the engine.
The rail-off idle is why the efficiency map subtracts a flat zero for the engine baseline while the GPU and CPU baselines subtract a nonzero idle draw.

## Power scales with utilization

The idle rail reads zero and the dispatch floor draws about 0.9 watts, and from there the draw climbs with how much of the multiply array a workload keeps active.
A twenty-point sweep on an M1 Max, read at the root power sampler, traces the curve: a dispatch-floor matrix multiply draws about 0.9 watts, a compute-bound 1024 by 4096 fp16 matrix multiply about 4.3 watts, and the int8 form about 5.8 watts.
That is a roughly fivefold range set by utilization alone.
The draw is regime-dependent as well: a bandwidth-bound matrix multiply that mostly streams a large weight draws about 1.4 watts, a convolution about 1.3 watts because its lowering leaves part of the array idle, and an elementwise operation almost nothing on the engine rail.
Efficiency peaks at the fp16 compute optimum near 2.68 trillion operations per watt, about 0.37 picojoules per FLOP, and falls to about 22 picojoules per FLOP at the dispatch floor, which is a further reason to amortize work past the floor.
The engine exposes only the power reading and a binary on-or-off state even to the root sampler, with no frequency or voltage telemetry, so the draw is observable but the sequence of operating points behind it is not.

## Sustained load holds one clock state

The efficiency figures above are warmup measurements, so they leave open whether the engine holds its power state under minutes of continuous load.
A compute-bound probe answers it directly: a chain of eight 1024-square fp16 matrix multiplies run as one program in a low-overhead zero-copy loop, at about 2.11 milliseconds per call, roughly ten times the dispatch floor, so the loop is compute-bound.
Run continuously for 210 seconds and 98,092 calls on the M1, the throughput curve is flat.
The whole-run median is 2109.6 microseconds, the start bucket reads 2153 microseconds and the end bucket 205 seconds later reads 2143.
The per-five-second bucket medians vary by about 4 percent in a non-monotonic band that is sampling noise, not thermal decay.
The engine rail, sampled read-only at five-second cadence, pins at about 5.5 watts for the whole run, with start, middle, and end thirds reading 5530, 5522, and 5391 milliwatts, a drift under 3 percent.
Thermal pressure reads Nominal on every sample, and the engine never leaves its single clock state, with the steady draw set by the workload's utilization as the previous section describes.
The M1 engine does not throttle on this workload over 3.5 minutes: it runs at one clock state and stays there.

The behaviors that move first-call latency are at the boundaries of a burst, not in steady state, and [Table](#tbl:c10-phases) gives the warmup, steady-state, and idle re-wake costs the flat steady state hides.

| Phase | Behavior on the M1 |
| --- | --- |
| Warmup | About a three-call ramp: call 1 about 7.6 ms (3.6x steady), call 2 about 4.6 ms, call 3 within 13 percent of steady, then flat at about 2.15 ms. |
| Steady state | Flat at about 8.1 TFLOP/s and about 5.5 W for 3.5 minutes, p50 2110 microseconds, p99 only 36 percent over p50. |
| Sub-second idle | A modest first-call penalty of 1.2x to 1.7x, then the next call is back to steady. |
| Multi-second idle | The first call after a 5-second gap costs about 260 ms, roughly 123x the steady p50, then the very next call returns to about 2.1 ms. |

Table: Sustained-load phases on the M1, with the warmup and idle re-wake costs that the flat steady state hides. {#tbl:c10-phases}

The idle re-wake penalty is the consequence of the rail-off idle described above.
Once idle crosses a few seconds the compute sets gate fully off, and the first call after the gap pays a one-time cold re-wake of tens to hundreds of milliseconds before the engine returns to its steady state on the following call.
The cost falls on the first call of each active burst, not as a sustained slowdown.
Latency-sensitive code that must answer immediately after an idle gap should thus keep the engine warm with a sub-second dispatch cadence or a low-cost keep-alive call.

The engine rail sampled directly on the M1 confirms that average power tracks utilization.
Idle draws about 0 milliwatts.
A sustained batch-of-512 hot loop draws about 1.48 watts.
A single-call loop draws about 755 milliwatts, about half the hot-loop figure, because a single small call leaves the engine idle for most of the dispatch window.

## Compression as an energy control

Streaming a weight in a narrower format cuts energy per inference even though instantaneous power stays roughly flat.
On a bandwidth-bound 4096-inner matrix multiply on the M2, the engine draws between 2.06 and 2.59 watts across fp16, int8, int4, and sparse, with the sparse stream drawing the most watts.
The energy per inference still falls with the narrower format because latency falls faster than power rises, as [Table](#tbl:c10-formats) gives the power, latency, and energy per inference across fp16, int8, int4, and sparse.

| Weight format | Engine power | Latency | Energy per inference | Versus fp16 energy |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 2.18 W | 0.690 ms | 1.50 mJ | 1.00x |
| int8 | 2.06 W | 0.431 ms | 0.89 mJ | 0.59x |
| int4 | 2.29 W | 0.270 ms | 0.62 mJ | 0.41x |
| Sparse (about 32 percent dense) | 2.59 W | 0.330 ms | 0.86 mJ | 0.57x |

Table: Power, latency, and energy per inference across weight formats on a bandwidth-bound matrix multiply, M2. {#tbl:c10-formats}

An int4 weight stream reaches the same answer at 0.41 times the fp16 energy on this part, a 2.4 times energy reduction at equal work, with no efficiency-versus-latency tradeoff to weigh: the narrower format is faster and more efficient.

## Developer takeaway

The fp16 datapath is why the lead holds even where the GPU is faster on latency: a narrow multiply and a fixed-function pipeline move and compute far fewer bits per result than a general-purpose vector unit.
The engine thus spends less energy reaching the same answer wherever the math stays in its precision range.

## Estimating a layer's energy before it is built

A workload's efficiency follows from where the roofline locates it.
A layer that is compute-bound on the engine spends its energy in the fixed-function multiply array, which is the regime where the engine outruns the GPU on throughput per watt by the wide margins in the map above.
A layer pinned to the memory slope, or below the dispatch floor, spends its energy moving bytes or paying call overhead, where the efficiency lead narrows toward a tie.
The cost estimate locates the layer statically, so the energy regime can be read before any device is in hand.

The procedure estimates the layer, reads its bound, and locates it against the efficiency map: a compute-bound layer is where the engine is most efficient, a dispatch-bound one is not.

```python
# Estimate the energy of one layer before any hardware is in hand.
# A 3x3 stride-1 convolution, 256 channels, on a 56 by 56 feature map.

given layer L = conv_3x3(input = [1, 256, 56, 56], stride = 1)
given target chip = H13            # M1; peak and bandwidth come from its roofline

# 1. Count the work the layer does and the bytes it must move.
flops = total multiply_adds in L           # arithmetic operations
bytes = weight_bytes(L) + input_bytes(L) + output_bytes(L)

# 2. Locate the layer on the roofline of the chosen chip.
peak      = compute_peak(chip)             # operations per second the array can sustain
bandwidth = memory_bandwidth(chip)         # bytes per second from on-chip and DRAM
floor     = dispatch_floor(chip)           # fixed per-call overhead, about 0.23 ms on H13

compute_time = flops / peak                # time if the array is the limit
memory_time  = bytes / bandwidth           # time if moving bytes is the limit
latency      = max(compute_time, memory_time) + floor

if compute_time >= memory_time:  regime = "compute"     # array busy, most efficient regime
else:                            regime = "bandwidth"    # moving bytes, efficiency lead narrows
if latency is dominated by floor:  regime = "dispatch"  # call overhead, leads converge

# 3. Energy = power drawn in that regime, times how long the layer runs.
power  = sustained_power(chip, regime)     # e.g. about 1.78 W compute-bound on H13
energy = power * latency

# 4. Amortized versus per-call: a layer called once pays the full floor,
#    a layer called many times spreads that fixed floor across the calls.
energy_per_call_standalone = power * (max(compute_time, memory_time) + floor)
energy_per_call_amortized  = power * (max(compute_time, memory_time) + floor / number_of_calls)

return regime, latency, energy_per_call_standalone, energy_per_call_amortized
```

A layer that prints `compute` runs in the region where the convolution-stack figures hold, up to 2063 GFLOP/s per watt on the M1; a layer that prints `dispatch` or `bandwidth` falls toward the tie at the small-reduction and trivial-matmul regions.

## Reference: engine versus GPU efficiency constants

[Table](#tbl:c10-constants) collects the power and efficiency constants of this chapter with the silicon each was measured on.

| Quantity | Value | Silicon |
| --- | ---: | --- |
| Convolution-stack efficiency, engine | 2063 GFLOP/s/W | M1/H13 |
| Convolution-stack efficiency, GPU | 142 GFLOP/s/W | M1/H13 |
| Convolution-stack efficiency ratio | 14.5x | M1/H13 |
| Convolution-stack efficiency, engine | 2289 GFLOP/s/W | M5/H17s |
| Convolution-stack efficiency, GPU | 175 GFLOP/s/W | M5/H17s |
| Convolution-stack efficiency ratio | 13x | M5/H17s |
| Convolution-stack efficiency, engine | 2234.6 GFLOP/s/W | M2/H14 |
| Convolution-stack efficiency, GPU | 172.8 GFLOP/s/W | M2/H14 |
| Convolution-stack efficiency ratio | 12.9x | M2/H14 |
| Sustained 3x3 convolution efficiency | 643 GFLOP/s/W | M1/H13 |
| Engine power, sustained convolution | 1.78 W | M1/H13 |
| Engine power, sustained convolution | ~3.8 W | M2/H14 |
| Engine power, compute-bound GEMM | ~5.9 W | M2/H14 |
| Large 4096 GEMM, engine power | 4.4 W | M1/H13 |
| Large 4096 GEMM, GPU power | 32.5 W | M1/H13 |
| Large 4096 GEMM efficiency ratio | 4.0x | M1/H13 |
| Five-point stencil efficiency ratio (fused) | 49x | M5/H17s |
| Five-point stencil efficiency ratio (fused) | 47.8x | M2/H14 |
| ViT-B/16 forward energy, engine versus GPU | 67.9 against 714.3 mJ/inf, 10.5x | M2/H14 |
| ResNet-18 forward energy, engine versus GPU | 2.6 against 22.3 mJ/inf, 8.5x | M2/H14 |
| int4 weight stream energy versus fp16 | 0.41x at equal work | M2/H14 |
| Workload-class efficiency envelope | 2 to 14x | M1/H13 |
| Engine rail, upper bound on the measured classes | ~6 W | M2/H14 |
| Idle engine power | ~0 W (rail-off) | M1/H13 |
| Engine rail, sustained batch-of-512 hot loop | ~1.48 W | M1/H13 |
| Engine rail, single-call loop | ~755 mW | M1/H13 |
| Effective stream rate, M1 | ~10 GB/s | M1/H13 |
| Effective stream rate, M5 | ~57 GB/s | M5/H17s |

Table: The power and efficiency constants of this chapter, with the silicon each was measured on. {#tbl:c10-constants}
