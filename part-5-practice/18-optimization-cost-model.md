# 18. Optimization and the cost model

> The compiler turns a shape into a wall-time estimate through three stages: a cycle count, roofline, and fixed dispatch floor, with only the per-chip parameters changing across the family.
> On the M1 the model fits the five reference convolutions it was tuned against within plus or minus 17 percent at a peak of 3.25 fp16 TFLOP/s, bandwidth of 9.0 GB/s, and 0.23 ms floor.
> Across a broader sweep the median error is about 31 percent, so the estimate is an ordinal placement tool rather than an absolute-latency oracle.
> The same model estimates latency for any of the 28 targets without the chip in hand, since bandwidth scales by core count and the floor by clock.
> The autotuner ranks equivalent rewrites by this estimate and preserves accuracy by default, with a 3.7 to 5.3 times gain on attention blocks at cosine similarity 1.0.

The compiler estimates how long a layer will take before it runs.
The same model drives an ahead-of-dispatch latency estimate for any chip in the family and a deterministic autotuner that ranks equivalent graph rewrites.

## Three stages of the model

The first stage estimates compute cycles from the operation and its dimensions.
For a convolution the cycle count is the input-channel passes times the kernel volume times the output spatial extent, divided by the active compute units, and sparsity scales it down by the fraction of zero weights.
Pooling and elementwise operations use their own cycle variants of the same form.

The second stage applies a roofline.
Compute time is the cycle count divided by the throughput rate, memory time is the operand bytes divided by the bandwidth, and the layer latency is the larger of the two.
A layer whose memory time exceeds its compute time is bandwidth-bound; the reverse is compute-bound.

The third stage converts to wall time.
The model runs at a clock near 0.8 times the maximum frequency, scales the rates by a per-chip efficiency curve, and adds a fixed overhead for the per-input transfer setup plus a constant dispatch cost.
The result is a latency in microseconds.

$$t \approx \max\left(\frac{\mathrm{cycles}}{f},\ \frac{\mathrm{bytes}}{B}\right) + t_0$$

The throughput is set by the compute-unit geometry and the efficiency curve, $f = \mathrm{cores} \times 4 \times \mathrm{eff}(f) \times \mathrm{clock}$, and the bandwidth $B$ is the DMA rate the chip sustains across the operand stream.
On the M1 the model fits the measured convolution latencies with a peak of 3.25 fp16 TFLOP/s, bandwidth of 9.0 GB/s, and dispatch overhead $t_0$ of 0.23 ms, all five reference convolutions within plus or minus 17 percent.

## Per-chip parameters

Only the parameter values change from chip to chip; the three-stage structure is identical across the family.
The compute-unit count, per-cycle divisor, clock range, and efficiency curve are read from the hardware abstraction table for each target, the table chapter 24 decodes.
[Table](#tbl:c18-chip-params) gives the source and the fitted M1 and M5 values for each parameter side by side.

| Quantity | Source | M1/H13 | M5/H17s |
| --- | --- | --- | --- |
| Compute units | core-count field | 4 | 16 |
| Cycle divisor | divisor field | 64 | 64 |
| Clock range | DVFS table | 0.40 to 1.43 GHz | 0.74 to 2.36 GHz |
| Efficiency | frequency curve | 1.0 | about 0.84 |
| Fitted peak | silicon anchor | 3.25 fp16 TFLOP/s | 8.9 fp16 TFLOP/s |
| Fitted bandwidth | silicon anchor | 9.0 GB/s | 57 GB/s |
| Dispatch floor | silicon anchor | 0.23 ms | 0.11 ms |

Table: The cost-model parameters that change from chip to chip, with their source and the fitted values for the M1 and M5. {#tbl:c18-chip-params}

The M1 model takes efficiency as 1.0, meaning it sustains peak at any clock; later generations derate to about 0.84 because the array cannot hold the peak multiply rate at the higher clock ceiling.
The compute units scale 4 to 16 to 32 to 64 across the die variants, and the clock ceiling rose from 1.43 GHz to 2.36 GHz, which together put the M5 theoretical peak near 5.5 times the M1. The fitted peaks in the table scale more conservatively, about 2.7 times, from 3.25 to 8.9 fp16 TFLOP/s, since the cost model anchors to the rate the array sustains rather than the theoretical product.

Bandwidth scales with the compute-unit count rather than the clock.
The M5 streams the same bandwidth-bound model at 57 GB/s against the M1's 9.0 GB/s, close to the 16-to-4 core ratio, so the per-chip anchor sets bandwidth by core count and the dispatch floor by clock.
The 9.0 GB/s here is the cost model's jointly calibrated effective fit, and it is not the 51 GB/s single-row saturating weight-stream of chapter 9 nor the roughly 40 GB/s broad-shape effective rate, which measure different things.
A model anchored to the M1 alone over-predicts M5 latency by a mean of 99 percent: clock-scaling the M1 bandwidth yields about 15 GB/s, which the true 57 GB/s rate exceeds by about 3.8 times.
Substituting the M5 measured bandwidth, floor, and peak brings four of the five reference convolutions within 13 percent.
[Listing](#lst:c18-estimate-latency) evaluates the three-stage model against per-chip anchors, taking the larger of the compute and memory terms and adding the fixed dispatch floor.

```python
def estimate_latency(cycles, op_bytes, chip):
    # Stage 1 input: cycles = input-channel passes * kernel volume *
    #                output spatial extent / active compute units.
    # Stage 2: roofline picks the binding term.
    compute_time = cycles / chip["peak_flops"]   # f: cores * 4 * eff * clock
    memory_time  = op_bytes / chip["bandwidth"]  # B: DMA rate the chip sustains
    # Stage 3: add the fixed per-eval dispatch floor.
    return max(compute_time, memory_time) + chip["dispatch_floor"]

# Fitted M1/H13 anchors: peak 3.25 fp16 TFLOP/s, B 9.0 GB/s, floor 0.23 ms.
m1 = {"peak_flops": 3.25e12, "bandwidth": 9.0e9, "dispatch_floor": 0.23e-3}
# A small operation collapses to the dispatch floor; a large one tracks
# whichever of compute_time and memory_time binds.
```

Listing: The three-stage cost model evaluated against per-chip anchors, taking the larger of the compute and memory terms plus the dispatch floor. {#lst:c18-estimate-latency}

## Reading the model before dispatch

The model returns a per-chip latency estimate for an output graph without running it, which locates a layer against the roofline of chapter 9 ahead of time.
The estimate is the same three-stage number described above, evaluated for a named target.
For chips that cannot be run locally it is the only available latency figure, since the structure is identical across all 28 targets and only the per-chip parameters differ.

Two readings of the estimate matter most.
A small operation collapses to the dispatch floor, and the floor is the optimization target: an operation whose compute term is under 0.23 ms on the M1 gains nothing from a faster array, so the work must grow to clear the floor.
A large operation tracks the binding roofline term, and the question is whether the layer is compute-bound or bandwidth-bound, which decides whether shape or streaming is the control.

Fusion removes floors and intermediate round-trips.
A network run as separate dispatches pays the floor on every one and copies each intermediate back to the host and forward again; the same network fused into one program pays the floor once and keeps the intermediates resident.
The model accounts for this: fusing operations removes their separate $t_0$ terms and the operand bytes of the eliminated round-trips, which is why a network is compiled as one program rather than a sequence of small ones.

## What the optimizer does

The optimizer ranks equivalent rewrites of a graph using an op-agnostic cost estimate and a deterministic autotuner.
The cost estimate is the analytic roofline-plus-floor number, applied uniformly to every operation rather than holding a hand-tuned constant per operation type.
The autotuner enumerates rewrites that compute the same result, estimates each, caches the measured outcomes, and selects the lowest deterministically, so the same graph yields the same choice on every run.

The rewrites preserve accuracy by default.
A route rewrite that decomposes attention into a fused form is selected only when it computes the same result to within tolerance, and a rewrite that trades precision, such as an integer-quantized weight stream, is taken only under an explicit tolerance and a speedup margin.
The default never changes the numerical result of the graph.

The reported gains follow the roofline.
A fused attention route rewrite is 3.7 to 5.3 times faster on attention blocks and on a full vision transformer at cosine similarity 1.0, because it removes per-dispatch floors and intermediate round-trips rather than changing the arithmetic.
The optimizer's gate is the test corpus, which holds the cross-chip latency table as a regression so a change to the model cannot silently move an estimate.

## Estimating latency and tuning a graph

The estimate locates a graph against any target's roofline before dispatch, and the autotuner selects the lowest-latency equivalent rewrite under an accuracy bound.
The estimate returns the three-stage latency for a named target with no device in hand, and the tune step ranks accuracy-preserving rewrites by that same estimate.
[Listing](#lst:c18-estimate-tune) estimates a graph for a named target with no hardware in hand, then enumerates equivalent rewrites and keeps the lowest-latency one that holds the accuracy tolerance.

```python
# Estimate latency for a named target with no hardware in hand, then tune.
# The three-stage estimate: compute_time = cycles / f, memory_time = bytes / B,
# latency = max(compute_time, memory_time) + dispatch_floor (t0).

target = H17s                                 # per-chip f, B, t0 read from the hardware table

function estimate(graph, target):
    cycles = sum over ops of compute_cycles(op)        # stage 1: per-op cycle count
    bytes  = sum over ops of operand_bytes(op)
    compute_time = cycles / f(target)                  # stage 2: roofline
    memory_time  = bytes  / B(target)
    latency = max(compute_time, memory_time) + t0(target)   # stage 3: add dispatch floor
    if   latency near t0(target):       bound = "dispatch"
    elif memory_time > compute_time:    bound = "bandwidth"
    else:                               bound = "compute"
    return latency, bound

latency, bound = estimate(G, target)
# If bound == "dispatch": batch or fuse until the compute term clears the floor;
# no faster array helps a layer pinned to the dispatch floor.

# Tune: enumerate equivalent rewrites, estimate each, keep the lowest that holds the tolerance.
best := G                                      # the unmodified graph
best_latency := latency
for each rewrite R that computes the same result as G:
    if max_abs_difference(R, G) <= tolerance:  # accuracy-preserving only (tolerance = 0 by default)
        latency_R, _ = estimate(R, target)
        if latency_R < best_latency:
            best := R
            best_latency := latency_R
# A fused-attention route rewrite is 3.7 to 5.3 times faster on attention blocks at cosine 1.0.
# Note: attention has no analytic cost form, so its estimate is absent and must be timed on device.
```

Listing: Estimating a graph's latency for a named target with no hardware, then ranking accuracy-preserving rewrites by the same estimate. {#lst:c18-estimate-tune}

A graph whose estimate prints `dispatch` is batched or fused until the compute term clears the floor, since no faster array helps a layer pinned to the 0.23 ms floor on the M1.

## Two cost models and their coefficients

There are two cost models in the toolchain, and they do different jobs.
The model above is the intra-engine analytic model the compiler holds, the cycles-to-roofline-to-wall-time chain, fit to silicon for a per-chip latency estimate.
A second model decides backend placement: whether each operation runs on the engine, central processor, or graphics processor, and it is in the segmenter rather than the engine compiler.
The placement model costs an operation as the same roofline form, `max(flops/peak, bytes/bandwidth) + launch`, but with coarse abstract anchors rather than silicon-fit ones.

A live read of the compiler shows how the intra-engine model scores one layer.
A layer's cycles are the larger of the core compute time and the direct-memory-access and L2 transfer time, plus the dependency stalls, computed in `ZinEnginePerf::ComputeRunTime`.
The layer total is the execute cycles plus the overhead cycles.
The tiling choice is the argument that minimizes the split cost against the unsplit cost, searched once and cached.
`CalculateExeCycles` reads a precomputed double at layer offset `+0x1e0` and saturates at 65535, the width of the 16-bit task-descriptor field.

This intra-engine per-layer cost model is distinct from the placement cost model that decides engine versus central processor or graphics processor.
The two answer different questions at different layers: the per-layer model sizes and tiles work already bound for the engine, while the placement model orders the three backends.

The live measurement validates the analytic roofline.
The measured full-call floor of about 190 microseconds is about 16 percent below the frontend's analytic per-chip anchor of 220 microseconds, inside the model's stated plus-or-minus 17 percent fit.

The placement-model anchors are per-backend, not per-chip.
[Table](#tbl:c18-backend-anchors) gives the abstract roofline peak and bandwidth each backend is charged against, read from the segmenter cost functions.

| Backend | Peak (GFLOP/s) | Bandwidth (GB/s) |
| --- | ---: | ---: |
| Engine | 800 | 50 |
| Graphics processor | 120 | 40 |
| Central processor | 20 | 10 |

Table: The abstract per-backend roofline anchors the placement model uses to order the three backends, read from the segmenter cost functions. {#tbl:c18-backend-anchors}

These anchors are deliberately coarse.
The 800 GFLOP/s engine peak is about four times under the M1 silicon fit of 3.25 fp16 TFLOP/s, and the 50 GB/s engine bandwidth is above the 9 GB/s effective fit, so the two errors run in opposite directions and the net placement order survives.
The job of these numbers is to separate the engine from the graphics processor by 6.7 times and from the central processor by 40 times, a separation that holds under any single miscalibration.
Absolute latency accuracy is not their responsibility; it is in a learned per-operation layer described next.

The placement model also charges a launch cost on every segment and a transfer cost on every backend crossing, which is why the segmenter prefers long single-backend runs, as [table](#tbl:c18-penalties) gives in the model's relative units.

| Penalty | To the engine | To another backend |
| --- | ---: | ---: |
| Launch cost per segment | 0.05 | 0.10 |
| Transfer cost across backends | 0.09 | 0.23 |

Table: The fixed launch and transfer penalties the placement model charges, in the model's relative units. {#tbl:c18-penalties}

A transfer between two operations on the same backend costs zero, so a long run kept on the engine pays neither the per-segment launch nor the per-crossing transfer.

## Learned per-operation leaves

Above the coarse roofline the placement model holds a learned per-operation layer: 322 regression trees, one set per platform class and compute path, that map the coarse roofline ratio onto a calibrated cost in nanoseconds.
The trees split on a normalized feature built from the operation flops and bytes, and the leaf is the predicted cost.
The engine trees are the small roofline-form ones, and the cost they return is monotone in operation cost: elementwise below pooling below linear below convolution below transposed convolution.
[Table](#tbl:c18-tree-leaves) gives representative engine cost-tree leaves, spanning a constant-leaf elementwise operation to the largest transposed-convolution leaf.

| Operation (engine) | Tree form | Leaf cost |
| --- | --- | ---: |
| `relu` | constant leaf | 50 ns |
| `add` | constant leaf | 26.5 to 36 ns |
| `mul` | constant leaf | 40 to 80 ns |
| `reduce_sum` | constant leaf | 22 ns |
| `max_pool` | single-threshold leaf | 26 to 29 ns |
| `conv` (main tree) | seven-split learned tree | 417.7 ns to 8609.6 ns |
| `matmul` | three-threshold tree | 116.9 ns to 66408.3 ns |
| `linear` | three-threshold tree | 67 ns to 14.7 microseconds |
| `conv_transpose` | three-threshold tree | 3.5 to 37.8 microseconds |

Table: Representative learned engine cost-tree leaves, in the model's nanosecond output. {#tbl:c18-tree-leaves}

Three operations have no engine tree at all, because the placement model never puts them on the engine: gather, the recurrent cell, and the argument-maximum reduction are central-processor or graphics-processor placements only.

The learned leaves and the silicon-fit analytic anchors reach calibrated accuracy by different routes.
The analytic model above replaces the coarse anchors with silicon-fit ones and brings all five reference convolutions within plus or minus 17 percent.
The placement model keeps the coarse anchors as a binning feature and bolts a learned per-operation tree on top.
The convolution case shows both against silicon.
[Table](#tbl:c18-conv-compare) sets the coarse placement roofline, the silicon-anchored analytic estimate, and the measured M1 latency against each other for four convolutions.

| Convolution | Coarse roofline only | Analytic, h13 anchors | Measured, M1 |
| --- | ---: | ---: | ---: |
| 3x3 C256 to 256 at 28 | 1156 microseconds | 505 microseconds | 507 microseconds |
| 1x1 C512 to 512 at 32 | 671 microseconds | 511 microseconds | 444 microseconds |
| 1x1 C1024 to 1024 at 16 | 671 microseconds | 569 microseconds | 686 microseconds |
| 1x1 C2048 to 2048 at 8 | 671 microseconds | 1210 microseconds | 1047 microseconds |

Table: Coarse roofline, anchored estimate, and measured M1 latency for four convolutions. {#tbl:c18-conv-compare}

The coarse roofline cannot even order these four: it ties the three one-by-one convolutions and misclassifies the memory-bound deep-narrow C2048 at 8 as compute-bound, predicting it faster than it runs.
The anchored analytic estimate reproduces the non-obvious silicon result that the deep-narrow C2048 at 8, with about a twentieth the multiply-accumulate count, runs about twice as slowly as the compute-bound 3x3 C256 at 28.
This is because the former is bandwidth-bound at arithmetic intensity 60 and the latter is compute-bound at 466.

## Fidelity the estimate actually has

The plus-or-minus 17 percent fit is a property of the five reference convolutions the M1 anchors were tuned against, not of the estimate at large.
A broad sweep of 68 graphs across eight operation families, each estimated by the cost model for the `h13` target and timed against the on-device per-call latency, locates the median absolute error at about 31 percent.
Only 11 of the 68 shapes are inside plus or minus 17 percent.
The estimate is sound as an ordinal placement tool rather than as an absolute-latency oracle: it identifies the binding roofline term and orders shapes correctly, so the backend choice and the relative ranking the optimizer needs survive the error.
The estimate reads as an approximate figure and a ranking, not a calibrated wall-time prediction outside the convolution shapes it was fit on.
The error is directional and concentrated in the bandwidth-bound regime, where the 9.0 GB/s anchor undershoots the roughly 40 GB/s effective rate the engine sustains and the estimate over-predicts the large-weight shapes by several times.

Attention has no estimate at all.
The cost model returns no estimate for any scaled-dot-product-attention graph, because attention compiles to a segmented plan whose native sub-programs the analytic model has no cost form for.
This is a coverage gap rather than an accuracy figure: the estimate is absent for attention, so such a graph must be timed on device.
