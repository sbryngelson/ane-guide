# 3. Numerics

> The datapath is fp16 end to end with a wide accumulator of fp32 class, so representable sums come back near exact.
> Convolution, matrix multiply, and normalization keep their precision; a cancellation-heavy step such as the transformer down-projection loses it to per-product fp16 input rounding, not to the accumulator, and needs a wider anchor off-engine.
> The fp16 ceiling is 65504, and a width-axis slice with a nonzero begin offset applies a fixed gain of sixteen, so a fill above 4094 overflows silently to infinity on the M1 and the A14.
> Activation lookup tables are accurate to about half a unit in the last place, but they coerce a NaN to a clamp value and have a small origin bias a bit-exact oracle must model.

The Apple Neural Engine computes in fp16 with a wide accumulator.
That property, stated in chapter 1, decides which computations keep their precision on the engine and which lose it.

## fp16 datapath

The multiply array is fp16 end to end.
A fp16 value decodes from a sign bit, five-bit exponent, and ten-bit fraction as

$$x = (-1)^s \, 2^{e-15} \left(1 + \frac{m}{2^{10}}\right),$$

with a maximum finite magnitude of $65504$.
Inputs are fp16, weights are fp16, outputs are fp16.
The frontend accepts fp32, int32, and bf16 as type annotations, but the backend does not implement them, so those annotations do not reach the silicon as wider arithmetic.
Bias is the single exception in the descriptor: it may be held as fp32, but it is added in the fp16 datapath.
The datapath reconstructs compressed weights to fp16 before they reach the multiplier; it dequantizes an int8, int4, or palettized weight to fp16 on the way in, and the multiply that follows is the same fp16 multiply as for an uncompressed weight.
The dequantization is affine, $w = s \, (q - z)$, with the scale $s$ scalar or per-output-channel, and agrees with the documented conversion-tool quantization relation [AppleCoreMLTools].
On the M1 generation the quantization is symmetric, so $z = 0$ and $w = s \, q$; the inverse encode is $q = \mathrm{round}(x/s) + z$.

The order of rounding in one multiply-accumulate pass is fixed.
A weight is dequantized to fp16, the fp16 multiply runs, the products accumulate in the wide register, an optional per-channel scale and bias apply in fp16, an optional activation table applies, and the result stores to memory as fp16.
Two rounding points bracket the accumulation: the inputs round to fp16 going in, and the output rounds to fp16 going out.

## Wide accumulator

The running sum between those two rounding points is not fp16.
The input port rounds tiles to fp16 on the way in and the output port rounds to fp16 on the way out, but the reduction in between is held in a wide register of fp32 class on every ANE device.
The accumulator width is a fixed hardware property, not a per-program or per-chip setting.

Representable sums thus come back near exact.
A reduction of sixteen thousand ones is bit exact, where a naive fp16 running sum would stall near two thousand once the partial total passes the spacing of its own increments.
The spacing of fp16 at a partial of magnitude $p$ is $\mathrm{ulp}(p) = 2^{\lfloor \log_2 p \rfloor - 10}$, so once $p > 2048$ the spacing exceeds $1$ and an added unit increment falls below half a step and is swallowed.

The worked case that shows the accumulator is wider than fp16 yet supplied by fp16-rounded inputs is the sum of one value of 4096 followed by 1024 ones. The exact sum is

$$[4096] + [1]\times 1024 = 5120,$$

and the engine returns 5116, between the naive-fp16 result $4096$ and the exact total.
A naive fp16 running sum returns 4096, because each added one falls below half the spacing at 4096 and is swallowed.
The engine instead holds the 1024 ones that fp16 would drop, and the small deficit from the exact 5120 is the fp16 rounding of the single input tile that holds the 4096, not a narrow running sum.

The discriminating test is a cancellation triple.
A sum of a large value, its negation, and a one, repeated sixteen times near a magnitude of four thousand, returns all sixteen ones intact.
An fp16 running sum holding four thousand has a spacing of two there, so each one would round away.
The ones survive, so the accumulator is physically wider than fp16.
The first reduction stage groups input lanes into tiles of four, fp16-rounded, which then supply the one wide accumulator.
Swept across magnitudes, the survivor count is layout-independent: the same result whether the one precedes or follows the negation, so the hardware re-associates over a fixed lane lattice rather than the source order.
[Table](#tbl:c3-cancel-sweep) records the survivor count against the cancellation magnitude and the fp16 spacing there.

| Cancellation magnitude | fp16 spacing at that magnitude | Survivors of sixteen ones |
| ---: | ---: | ---: |
| 1024 | 1 | 16 |
| 3000 | 2 | 16 |
| 4090 | 2 | 16 |
| 4096 | 4 | 4 |
| 8000 | 4 | 4 |
| 16000 | 8 | 4 |
| 30000 | 16 | 4 |

Table: The cancellation-threshold survivor sweep, M1/H13 measured, which fixes the first-stage tile width at four. {#tbl:c3-cancel-sweep}

The survivor count holds at sixteen up to a magnitude of 4090, then drops to a hard floor of exactly four at 4096 and stays at four out to 30000.
The threshold is at 4096 because that is the first magnitude where the fp16 spacing reaches four, so a one sharing a four-lane tile with a rounded partial at or above 4096 falls below the in-tile rounding threshold and vanishes.
The flat floor of four is the signature of a four-lane tile: a three-element triple period beating against a four-lane tile leaves exactly four unaffected lanes out of sixteen.
A single one added next to one large resident value cannot expose the tile, because that value forces the whole wide partial onto its own coarse spacing and the one rounds away regardless of which tile it is in.

Both behaviors are reproducible on the engine by reducing a fp16 vector.
The cancellation probe routes a sum through a matmul against a ones vector so the reduction is in the wide accumulator, and the slice trigger exercises the width-axis crop gain, as [listing](#lst:c3-accum-probe) gives both probes.

```python
# Pseudocode for the engine's wide accumulator and the width-slice saturation.
# Each reduce_via_matmul(v) runs on the engine: v is fp16, summed against a
# fp16 ones-vector so the reduction accumulates in the wide (fp32-class) register;
# the result is read back as fp16.

# (a) Cancellation probe: a large value, its negation, and ones near the threshold.
#     A naive fp16 running sum at 4000 has ulp = 2 and swallows every one.
big = 4000.0
v = ([big, -big, 1.0] * 16)               # 16 ones survive the cancellation
assert reduce_via_matmul(v) == 16.0       # engine keeps all 16; fp16 loop -> 0

#     The accumulator is wide, but the inputs still round to fp16 going in:
v = [4096.0] + [1.0] * 1024               # one tile of 4096, then 1024 ones
assert reduce_via_matmul(v) == 5116.0     # between naive-fp16 4096 and exact 5120

# (b) Width-axis slice saturation: a nonzero begin offset applies a x16 crop gain.
#     The fp16 ceiling is 65504, so 4094*16 == 65504 passes but 4096*16 overflows.
def width_slice_with_offset(value):       # slice begin offset != 0 on the width axis
    return value * 16.0                   # the fixed crop-DMA gain, then fp16 store

assert width_slice_with_offset(4094.0) == 65504.0          # at the ceiling, finite
assert width_slice_with_offset(4096.0) == float("inf")     # 65536 > 65504 -> inf
```

Listing: Probes for the wide accumulator's cancellation behavior and the width-slice saturation that overflows to infinity above the fp16 ceiling. {#lst:c3-accum-probe}

## Compressed weights and the streaming gate

The datapath reconstructs a compressed weight to fp16 before it reaches the multiply array, so the arithmetic is fp16 regardless of the storage format.
What differs by format and by chip is whether the compressed bytes stream to the engine compressed and decompress on the way in, or fold to dense fp16 in memory first.
A format that streams moves fewer bytes across the DRAM boundary; a format that folds yields the storage saving but not the bandwidth.
[Table](#tbl:c3-weight-stream) marks which formats stream and which fold across three chip generations.

| Format | M1 / A13 | A14 / M2 | A16 / M5 |
| --- | --- | --- | --- |
| int4 palette lookup table | stream | stream | stream |
| int8 affine | fold | stream | stream |
| sparse | stream | stream | stream |
| blockwise | fold | fold | stream |

Table: Which compressed-weight formats stream natively versus fold to dense fp16, by generation, M1 measured with A14 and M5 rows from the per-format gate. {#tbl:c3-weight-stream}

On the M1 the int4 palette form streams natively, measured at about 2.37 times the bandwidth of the dense weight, and a sparse weight with at least half its values zero streams as a one-bit keep-mask with the packed fp16 nonzeros.
The int8 affine and blockwise forms fold to dense fp16 before the data-movement step.
The fold expands the int8 weight to a dense fp16 constant in DRAM before the multiply, so the bytes that cross the DRAM boundary are full-width fp16 and the layer gets no bandwidth gain on the M1.
The int8 fold is a stored-size saving only on the M1.
The weight is half the size on disk, but it is reconstructed to fp16 in DRAM before the data-movement step, so a weight-streaming-bound matmul moves the same bytes as fp16 and runs at the fp16 latency.
The int8 form first streams as int8 on the A14 and M2 generation, where it is dispatched as int8 and dequantized at the multiply port so it moves half the bytes of fp16.
A bandwidth-bound matmul reaches about 0.52 times the fp16 latency at a weight of 8192 by 8192.
The compressed-weight quantization adds error only at the input rounding: an int8 conv weight tracks the fp16 result at a cosine near 1.0, with a relative error near 0.6 percent against an fp32 reference, against about 0.02 percent for fp16.

## Where precision holds and where it is lost

Convolution, matrix multiply, and normalization keep their precision when their partial sums stay in fp16 range and their results are representable.
The wide accumulator holds the reduction, and the only quantization is the fp16 rounding of the inputs and the output.
Vision, audio, and encoder workloads are in this regime, and run on the engine without a precision penalty.

Precision is lost on cancellation-heavy steps.
The transformer decoder down-projection is the case that fails: a large positive and a large negative contribution nearly cancel, and the result is a small difference between two large numbers.
The loss does not come from the accumulator, which is wide enough to hold the partial.
It comes from the per-product fp16 rounding of the inputs and weights before they enter the accumulator.
Once the operands are quantized to fp16, the cancellation amplifies that quantization into the result.
A cancellation-heavy step has no fp16-safe form on the engine and needs a wider anchor computed on the CPU or the GPU.

## Activation functions

Lookup tables evaluate the nonlinear activations.
Identity and plain ReLU are not table-driven, since ReLU is a max, but sigmoid, tanh, gelu, swish, erf, exp, and the rest route through a piecewise-linear table of fixed knots.
The table is a 33-knot piecewise-linear curve, not a dense sample grid.
The input maps affinely onto the 32 segments between 33 knots, the bracketing segment evaluates as a slope times the input plus an intercept in fp16, and the value clamps to the end-knot asymptote past the table domain.
Accuracy comes from the piecewise fit and the per-function domain, not from sample density.
A user sigmoid does not lower to the plain sigmoid table.
It lowers to a high-precision sigmoid table by default, which pushes the domain to a wider range with a finer subdivision near the linear region, so a sigmoid-heavy or attention-gate-heavy model stays accurate under fp16.

The decoded table is accurate enough that it is not a meaningful error source.
On the standard set, measured on device, the worst absolute error is at the level of the fp16 storage floor: sigmoid 0.0034, tanh 0.0017, gelu 0.0059, each under 0.4 percent of the function range.
The on-device value matches the fp16-rounded exact function to about half a unit in the last place.
The table adds nothing measurable on top of fp16 storage rounding.
The exceptions are sin, cos, and atan, which have up to about 0.04 to 0.12 absolute error near the seams of their argument reduction; a model that evaluates trig directly near a magnitude of pi should fold the range upstream.

## Edge behavior a correctness oracle must model

Several edge behaviors of the fp16 datapath depart from a host IEEE reference, and a model that reproduces engine results bit for bit has to encode them.
The engine coerces a NaN to positive infinity at the input boundary, and never produces a NaN anywhere.
A NaN sent in and echoed through the identity $x + 0$ returns positive infinity, with the same bits as sending positive infinity directly, and every downstream op then behaves as if the value had been positive infinity.
A NaN into relu returns infinity, a NaN into sigmoid or tanh returns 1.0, a NaN into erf returns 1.0, and a NaN into exp returns infinity.
A NaN into the maximum of two values returns infinity, and a softmax of a lane holding a NaN puts all the mass on that lane.
The one case where a NaN is not bit-identical to positive infinity is the variance reduction inside layer normalization, where the NaN enters as a large finite magnitude rather than as literal infinity.
Rms normalization gives identical results for a NaN input and an infinity input.
An upstream NaN through any gate thus leaves the engine as a finite or infinite value and does not surface as a NaN to downstream code.

The engine flushes to positive zero all the indeterminate forms that produce a NaN under IEEE.
The infinity minus infinity case returns positive zero, the zero times infinity case returns positive zero, $\mathrm{sqrt}(-1)$ returns positive zero, $\mathrm{rsqrt}(-1)$ returns positive zero, and $\log(-1)$ returns positive zero.

The engine preserves denormals elementwise but flushes them inside the multiply-accumulate.
A fp16 denormal down to $2^{-24}$ (about $5.96 \times 10^{-8}$) echoes bit-exactly through $x + 0$ and $x \times 1$ on the elementwise path, and scales correctly, so the common assumption that the engine flushes denormals globally is false for that path.
A pair of denormals summed inside a matmul, with a representable denormal total, returns positive zero, so the flush-to-zero is a property of the accumulator stage and not of the datapath as a whole.
The M5 accumulator instead preserves denormals, so a denormal input and a product of two denormals both survive a matmul reduction unflushed.
The flush is thus an M1-generation property rather than a fixed engine behavior, and the M2 through M4 parts between them were not measured, so the generation at which it changed is unknown.

Signed zero loses its sign before a reciprocal or a reciprocal square root.
The reciprocal of negative zero returns positive infinity where IEEE returns negative infinity, and the reciprocal square root of negative zero returns positive infinity where IEEE returns negative infinity, while $\mathrm{sqrt}$ of negative zero stays positive zero per IEEE.
A negative zero echoes as positive zero through $x + 0$, so the engine drops the sign bit of a zero before the reciprocal path.

A few activation tables collapse a large input rather than tracking it: softplus and softsign of positive infinity return positive zero, where the values should be positive infinity and a unit magnitude.
The logarithm of positive zero returns a finite sentinel of $-45440$ rather than negative infinity.
These are properties of the table approximations, distinct from the NaN coercion above.

Softmax subtracts a hardware maximum before the exponential, so it does not overflow even when a raw exponential would.
A softmax of $[1000, 1, 2, 3]$ returns $[1, 0, 0, 0]$ and matches a wide reference, despite $\exp(1000)$ being far past the fp16 range, and a softmax of four equal values returns four quarters.
A bare $\exp$ with no max-subtraction overflows to infinity at an input near $11.094 = \ln(65504)$, so the stable route is the fused softmax rather than a hand-rolled exponential over a sum.

Rounding on the fp16 output grid is round half to even.
A tie at the midpoint between two representable fp16 values rounds to the value with an even last bit, not away from zero.
A partial of $2049$ at a grid spacing of $2$ rounds to $2048$, the even neighbor, and a trailing half above the $2048$ threshold rounds the $L + 0.5$ accumulation to even rather than up.
The M5 returns $2050$ for this case, rounding up where the M1 rounds to even.
Whether the wide accumulator presents a value just above $2049$ rather than an exact tie, or the rounding differs by generation, is unresolved.

Some activation tables have a small constant bias at the origin.
The decoded gelu table returns $-0.000543$ at $x = 0$ where the exact gelu is $0$, and the swish table returns $-0.001259$ at $x = 0$ where the exact swish is $0$.
The bias is below the fp16 storage floor and does not affect the per-op accuracy figures, but a bit-exact oracle has to hold it.

## Saturation hazard

The fp16 maximum finite value is 65504.
The compute datapath has no fp32 margin, so a value above that saturates to infinity silently.

The multiply-accumulate output stage saturates earlier than the storage format, at exactly $2^{15} = 32768$, half of the fp16 ceiling.
This is a different axis from the width-slice gain below: it is a property of the accumulator output port, and it fires on matrix multiply, linear, and any convolution that accumulates two or more taps, whatever the number of accumulation terms.
The threshold is pinned to the bit: the largest fp16 value below $2^{15}$, which is $32752$, passes through a linear, and the next fp16 value, $32768$, returns infinity.
It holds for $K = 1$, for $K = 2$, and for a two-channel convolution, so it tracks the would-be output magnitude at the accumulator port and not the count of accumulated terms.
An interior partial that exceeds $2^{15}$ overflows to infinity even when a later cancellation would have brought the final result back into range.
The paths that drive a dedicated reduction or a single elementwise multiply hold the full fp16 range instead.
A reduce-sum rounds to 65504 first and then overflows, an elementwise multiply overflows at the true fp16 limit near 65536, and a pointwise one-by-one convolution with a single input channel passes a fill of 60000.
This earlier accumulator ceiling is consistent with the multiply-accumulate datapath having about one bit less margin than the fp16 storage format on the M1, so the guard for a matrix multiply or a multi-tap convolution is to keep the output magnitude below $2^{15}$.

The second saturation a developer encounters is on the slice path.
A width-axis slice with a nonzero begin offset routes through a crop DMA that applies a fixed gain of sixteen.
A value at or below 4094 passes through bit exact, because $4094 \times 16 = 65504$ is the fp16 ceiling.
A value of 4095 or above becomes infinity, because it rounds to 4096 on the fp16 grid and $4096 \times 16 = 65536 > 65504$, past the ceiling.
The control case, a slice with a zero begin offset, is free of the saturation at the same fill values, so the trigger is the nonzero width-axis offset and not the magnitude alone.
The hazard is on the width axis only: a nonzero begin offset on the height, channel, or batch axis stays finite at the same fill values.
The saturation is measured on both the M1 generation (H13) and the A14 generation (H14), where a fill at width offset 4094 stays finite ($4094 \times 16 = 65504$) and a fill at 4096 overflows to infinity ($4096 \times 16 = 65536$).
The non-saturating route arrives on the A15 generation and later, so the hazard is not fixed by the immediate next family.
The guard is to avoid nonzero last-axis begin offsets on a width slice, or to route the offset onto a different axis.

## Determinism

For a fixed graph and a fixed input the M1 engine is bit-deterministic: the raw fp16 output bytes are identical across reruns, across an independent recompile, and across a fresh process.
The lowered program fixes the accumulation order, so there is no run-to-run drift to round away.

Re-executing one compiled program on the same input returns the same fp16 bytes every time, measured at zero units in the last place over 200 repeats for a matrix multiply, two-layer convolution, and long reduction.
The same holds over 50 repeats for softmax, reduce-mean, and a large matrix multiply.
Compiling the same graph twice at optimization level zero, the byte-identical lowering path, produces two programs whose outputs agree to the bit, including a real-tiled $[128, 1024] @ [1024, 1024]$ matrix multiply.
Running the same graph in a fresh subprocess, with a new dispatch client, returns the same output digest as the in-process run, so daemon and process state do not perturb the result.

The result is also independent of batch size and batch position.
A row computed alone is bit-identical to the same row computed inside a batch, and the same row placed at every position of a batch of sixteen gives sixteen identical outputs.
Row zero is invariant across batch sizes of one, two, eight, thirty-two, and one hundred twenty-eight.
The engine computes each batch element identically regardless of its neighbors.

The one thing that changes the bits is changing the math.
Writing a graph with a different association order, $(a + b) + c$ against $a + (b + c)$, gives different fp16 rounding.
About 31 percent of the elements differ, by about one fp16 unit in the last place at that magnitude, with a maximum absolute difference near $7.8 \times 10^{-3}$.
Each ordering is itself perfectly reproducible, so this is ordinary fp16 non-associativity of the graph as written and not hardware nondeterminism.
Two high-level ops that lower to the same kernel agree to the bit.
A dot product written as a matrix multiply and the same dot product written as an elementwise multiply followed by a reduce-sum return identical bytes here, because the compiler lowered both to the same accumulation order.
Reassociation bites only when the graph dictates a different order, so the engine is safe to treat as reproducible, and the only nondeterminism source is the accumulation order the graph chooses, never the silicon.

## Keeping a graph inside the fp16 envelope

Two numeric decisions belong before a graph is compiled.
A developer anchors a cancellation-heavy reduction off-engine, on the CPU or the GPU, because it has no fp16-safe form on the datapath.
A width-axis slice avoids a nonzero begin offset, because the fixed gain of sixteen overflows a fill above 4094 to infinity.

The cancellation-heavy step routes to a wider unit, and the rest stays on the engine, as [listing](#lst:c3-safe-graph) works the three rules through one graph.

```python
# Keep a graph numerically safe BEFORE compiling it, by applying three rules
# to every reduction and every cropped tile in the graph.

# fp16 limits the whole datapath uses:
fp16_max         = 65504        # largest finite fp16 value; above this -> infinity
accum_out_max    = 32768        # multiply-accumulate output port saturates here
width_slice_gain = 16           # a width-axis crop with a nonzero begin offset
                                #   multiplies its values by this fixed gain

# RULE 1: route a reduction through a matmul against a ones-vector, so the
#         sum accumulates in the wide (fp32-class) accumulator and stays near exact.
#         A plain elementwise running sum rounds at every step in narrow fp16.
function safe_reduce_sum(vector v):                  # v is fp16
    ones = vector_of_ones(length(v))                 # fp16, same length as v
    return matmul(v, ones)                           # one dot product, wide accumulator

# RULE 2: a cancellation-heavy step (a large positive nearly canceling a large
#         negative) has no fp16-safe form on the engine, because the operands
#         were already rounded to fp16 before the subtract. Scale it up so the
#         small difference clears the fp16 grid, OR split it out to a wider
#         unit (CPU or GPU) and feed the result back as a graph input.
function place_cancellation_step(step):
    if is_cancellation_heavy(step):
        return compute_off_engine(step)              # wider anchor, fed back as input
    else:
        return keep_on_engine(step)                  # representable sums stay near exact

# RULE 3: keep every tile value under the limit that applies to its path,
#         so nothing saturates silently to infinity mid-graph.
function check_tile_value(value, path):
    if path == "matmul_or_multitap_conv":
        require value <= accum_out_max               # output-port ceiling, ~half fp16_max
    if path == "width_slice_with_nonzero_offset":
        require value * width_slice_gain <= fp16_max # 4094*16 == 65504 passes; 4096*16 overflows
    otherwise:
        require value <= fp16_max                    # plain elementwise / single reduction

# Assemble the safe graph: representable conv/matmul/norm stay on the engine,
# reductions go through safe_reduce_sum, the cancellation step is anchored wide.
graph G:
    input  x                                         # fp16 activations
    feats  = conv(x, weights = W)                    # representable sums, stays on engine
    pooled = safe_reduce_sum(feats)                  # RULE 1: wide-accumulator reduction
    output pooled
program P = compile(G, target = H13)                 # fp16 datapath, wide accumulator
result = dispatch(P, features)
# The cancellation-heavy down-projection (RULE 2) is computed off-engine and
# fed back as an input, never lowered into this graph.
```

Listing: The three rules that keep a graph inside the fp16 envelope before compile. {#lst:c3-safe-graph}

## Reference: fp16 numeric constants

[Table](#tbl:c3-constants) collects the fp16 numeric constants, the saturation thresholds, and the activation-table error figures.

| Constant | Value |
| --- | ---: |
| fp16 maximum finite magnitude | 65504 |
| Multiply-accumulate output ceiling | 32768 |
| Width-slice crop-DMA gain | 16 |
| Width-slice finite fill ceiling | 4094 |
| Width-slice overflow fill | 4096 |
| Wide-accumulator bit-exact reduction | 16000 ones |
| Worked sum, $[4096] + [1] \times 1024$ | 5116 |
| First reduction-stage tile width | 4 lanes |
| Sigmoid worst absolute error | 0.0034 |
| Tanh worst absolute error | 0.0017 |
| Gelu worst absolute error | 0.0059 |
| Gelu origin bias at $x = 0$ | $-0.000543$ |
| Swish origin bias at $x = 0$ | $-0.001259$ |
| Trig seam absolute error (sin, cos, atan) | 0.04 to 0.12 |
| Activation table knot count | 33 |
| Sigmoid table domain clamp | $[-9.938,\ +8.320]$ |
| Exp input where output first reaches infinity | 11.094 |
| Square input where output first reaches infinity | 256 |
| int4 palette native stream speedup, M1 | 2.37x |
| int8 fold on M1 (stored-size saving, no stream gain) | 1.0x fp16 latency |
| int8 weight-stream latency, A14/M2 (8192 weight) | 0.52x fp16 |

Table: The fp16 numeric constants, M1/H13 measured, with the saturation reproduced on A14/H14. {#tbl:c3-constants}
