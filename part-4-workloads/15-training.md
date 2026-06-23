# 15. Training on the engine

> The engine has no backward operation, yet a full forward, backward, and optimizer loop runs as ordinary inference-style graph operations with the optimizer state resident across steps.
> The registered gradient set matches the closed form to a cosine of $1.0000$, so transformers, normalization-based convolutional networks, and gated linear networks all train end to end.
> A small convolutional network trains to a final test accuracy of $0.9080$ on the M1 and $0.9070$ on the M5 at a loss scale of 1024, a difference of one test sample in one thousand.
> The conv weight-gradient saturates above 4094 on the width axis on the M1 and M2, so a preflight check guards that one numeric edge.

There is no backward operation, no gradient layer, and no adjoint primitive in the compiler's operation catalog.
A network still trains end to end on the engine, optimizer step included, because the forward pass, backward pass, and parameter update are all expressed as ordinary inference-style graph operations the engine already runs.
This chapter states how that expression works, which gradients are numerically trustworthy, where one path diverges in fp16, and why the result is a capability rather than a speed claim.

## A backward pass built from forward operations

A search of the compiler binary for an engine-native gradient layer returns nothing.
Every gradient operation held in the shared compiler belongs to the graphics-processor dialect, is costed by the graphics-processor cost model, and executes there rather than on the engine.
The convolution data-gradient and weight-gradient, the max-pool and average-pool gradients, normalization, recurrent, top-k, pad, strided-slice, and resize gradients are all graphics-processor operations.
At the operation level the engine is inference-only.

The engine-native convolution family is the forward convolution and a single-channel cross-correlation.
The forward convolution has a transpose mode, so the data-gradient, which is a transposed convolution, has a native operation.
The weight-gradient has no native operation, because the only cross-correlation is single-channel and cannot serve the multi-channel correlation a weight-gradient requires.
The weight-gradient thus lowers through a patch-extraction expansion into a matrix multiply, which is why a trainable convolution holds its weight as a graph input rather than a folded constant and why its backward cost grows with the minibatch.

The public on-device update path meets the same boundary [AppleCoreML].
When the system framework fine-tunes a model on device, the forward pass may run on the engine, but the backward pass runs on the graphics processor or the host through those gradient operations.

Training the whole loop on the engine thus means not calling any backward operation.
The forward pass is a graph of engine operations.
The backward pass is a second graph, built by the host from the same forward operations and their analytic gradients, that computes cotangents through the network.
The optimizer update is a third graph that combines the gradients with the parameter and optimizer state.
All three are inference-style graphs of operations the compiler already accepts, so all three compile and dispatch on the direct route from chapter 6.

A trainable weight is a graph input, not an embedded constant.
The forward graph reads the weight from a bound buffer rather than from a folded constant tensor, so the update graph can write a new value into that buffer between steps without recompiling.
This is what lets the optimizer step run on the engine: the weight is an operand, and an operand can be both read by the forward pass and written by the update.

## Gradient vocabulary and its correctness

A gradient is built by composing each forward operation with its vector-Jacobian product.
The registered set was checked against closed-form derivatives by forming a linear loss $L = \sum_i (\mathrm{op}(x)_i \cdot w_i)$, whose exact gradient is $w \cdot \mathrm{op}'(x)$, and comparing the engine-computed gradient to that reference.

The core set matches to a cosine of $1.0000$ against the closed form, at the fp16 error level.
That set covers the activations `relu`, `sigmoid`, `tanh`, and `gelu`, the elementwise `mul`, `add`, `sub`, `square`, the linear `matmul` and batched `matmul`, `softmax`, `conv`, `avg_pool`, `max_pool`, the reductions `reduce_sum` and `reduce_mean`, and the shape operations `transpose`, `reshape`, `flatten`, `slice`, and `concat`.
The four normalization layers `layer_norm`, `rms_norm`, `group_norm`, and `l2_norm`, the `silu` activation, and a set of unary math operations including `exp`, `sqrt`, `rsqrt`, `log`, `erf`, and `cos` were added later and verified against a finite-difference reference at the same cosine.
The normalization gradients re-inject the per-channel scale as a supplied value-input, since the engine has no constant-tensor operation to hold it.
[Table](#tbl:c15-gradients) groups the registered operations by kind, from the activations through the reductions to the normalization layers.

| Group | Operations with a registered gradient |
| --- | --- |
| Activations | `relu`, `sigmoid`, `tanh`, `gelu`, `silu` |
| Elementwise | `mul`, `add`, `sub`, `square` |
| Linear | `matmul`, batched `matmul`, `softmax`, `conv`, `avg_pool`, `max_pool` |
| Reductions | `reduce_sum`, `reduce_mean` |
| Shape | `transpose`, `reshape`, `flatten`, `slice`, `concat` |
| Normalization | `layer_norm`, `rms_norm`, `group_norm`, `l2_norm` |
| Unary math | `exp`, `sqrt`, `rsqrt`, `log`, `erf`, `cos` |

Table: The registered gradient vocabulary, each verified to a cosine of $1.0000$ against the closed form or a finite-difference reference at the fp16 error level. {#tbl:c15-gradients}

With the normalization gradients present, a transformer block, normalization-based convolutional network, and gated linear network all train end to end on the engine, where previously only a plain multilayer network did.
Some operations have a forward but no registered gradient, among them `cumsum`, `amax`, `amin`, and several parametric activations.
A model that uses one of those compiles and runs its forward pass, then the backward construction raises an explicit error rather than producing a silently wrong gradient.
The gap there is coverage, not correctness.

## A resident-state training step

The optimizer state stays resident on the engine across steps, through the buffer-aliasing mechanism of chapter 2.
The first and second moments of an adaptive optimizer, along with the weights themselves, persist as buffers in the engine working set from one dispatch to the next.
The host sends only the per-step minibatch and the scalar learning rate, and reads the weight buffers back at a checkpoint.
The large held tensors never cross the host boundary on every step.

One dispatch advances the network by one optimizer step, as [listing](#lst:c15-step) shows: forward, backward, and update in a single submitted graph.

```python
# Resident buffers, allocated once and kept in the engine working set:
#   W           trainable weights        (read by forward, written by update)
#   M, V        optimizer moments        (read and written by update)
# Per-step host inputs: minibatch (x, y), learning rate lr_t

for step in range(num_steps):
    # all three stages are one engine graph, one dispatch
    logits = forward(W, x)                 # inference-style ops
    g      = backward(W, x, y, logits)     # vjp graph, no native backward op
    M      = beta1 * M + (1 - beta1) * g           # update, on engine
    V      = beta2 * V + (1 - beta2) * (g * g)
    W      = W - lr_t * M / (sqrt(V) + eps)        # writes resident W in place
    # M, V, W remain resident; host sends only (x, y, lr_t) next step
```

Listing: A resident-state training step run as one engine graph, with weights and optimizer moments resident across steps. {#lst:c15-step}

The adaptive update is

$$W_{t+1} = W_t - \eta_t \frac{M_t}{\sqrt{V_t} + \epsilon}, \quad M_t = \beta_1 M_{t-1} + (1 - \beta_1) g_t, \quad V_t = \beta_2 V_{t-1} + (1 - \beta_2) g_t^2$$

where $g_t$ is the gradient from the backward graph.
The host-supplied learning rate $\eta_t$ already absorbs any loss scaling, because the scale factor cancels in the ratio $M_t / \sqrt{V_t}$ and must not be divided out a second time.

A small convolutional network trains this way to a final test accuracy of $0.9080$.
On the M1 generation, the seeded handwritten-digit network reaches that accuracy after 300 steps, deterministic and reproducible to the digit across runs.

## Conv weight-gradient divergence

One gradient path diverges in fp16 on the M1 and the M2 generations.
The convolution is built from a patch-extraction step, a set of width-offset slices, followed by a matmul.
A nonzero-offset width slice on these generations saturates above 4094, the bound derived in chapter 3 and guarded in chapter 19.

The training-relevant consequence is that the weight-gradient runs the backward activations back through those same width-offset slices.
When the loss-scaled backward activations exceed about 4094, a few weight-gradient elements saturate to infinity and the gradient is corrupted.
The break is magnitude-gated and finite-to-infinity, not a small rounding error: at a fixed shape the path is exact at loss scale 384 and produces its first infinity at loss scale 512.
A larger input magnitude crosses the same threshold at a lower loss scale.

Two independent variables cross the threshold, the input magnitude and the loss scale, as [table](#tbl:c15-saturation) records across a sweep of both.

| Input scale | Loss scale | Result | Infinities |
| ---: | ---: | --- | ---: |
| 1 | 256 | free | 0 |
| 1 | 384 | free | 0 |
| 1 | 512 | saturates | 1 |
| 1 | 768 | saturates | 3 |
| 1 | 1024 | saturates | 8 |
| 4 | 256 | saturates | 36 |

Table: Conv weight-gradient saturation at a fixed shape, finite-to-infinity and magnitude-gated. {#tbl:c15-saturation}

A 1x1 convolution does not touch the slice path, because it has no width-offset patch slice.
The hazard is limited.
A convolution-first network sends the normalization-bounded input through the width-offset slices on the forward side, where values stay well below 4094 regardless of loss scale.
Its conv input-gradient, the path that would hold the loss-scaled gradient through the slice, is discarded because no layer precedes the first convolution.
The handwritten-digit network trains correctly at loss scales of 128, 1024, and 65536, with overlapping curves and final accuracies of $0.9070$, $0.9080$, and $0.9100$.
The residual risk is a network that pushes width-offset-slice values in the weight-gradient path past 4094, which a preflight check can flag on the M1 and M2 generations.
The M5 generation takes a different slice route and has no such saturation; the non-saturating route arrives on the A15 generation and later.

## Cross-generation parity

The same seeded network trains deterministically on both generations, with the data order and initialization fixed.
The two accuracy curves track to three decimals through all 300 steps and part by a single borderline prediction at the end, which [table](#tbl:c15-parity) traces step by step from initialization to step 300.

| Step | M1 test accuracy | M5 test accuracy |
| ---: | ---: | ---: |
| 0 | 0.0850 | 0.0850 |
| 50 | 0.8790 | 0.8810 |
| 100 | 0.9020 | 0.9020 |
| 150 | 0.9050 | 0.9030 |
| 200 | 0.9070 | 0.9070 |
| 250 | 0.9080 | 0.9070 |
| 300 | 0.9080 | 0.9070 |

Table: The seeded handwritten-digit network trained identically on both generations, deterministic and reproducible. {#tbl:c15-parity}

That one-sample gap is the end-to-end signature of the per-operation cross-generation fp16 difference, at most one unit in the last place per operation, accumulating across 300 steps until it flips one near-threshold logit.
The difference does not vanish, but it is far too small to affect the trained model, so training is portable across the two generations.

## A resident-state training loop

A training loop on the engine builds three graphs from the same forward operations: the forward pass, a backward graph from the registered gradients, and an adaptive update.
The weights and optimizer moments stay resident across steps, and the host sends only the per-step minibatch and the scalar learning rate, where the learning rate already absorbs any loss scaling.

The procedure of [listing](#lst:c15-loop) marks the weights as trainable graph inputs, keeps the optimizer state resident, and advances the network one optimizer step per dispatch.

```python
graph G:
    input   x : [B, 1, 28, 28] fp16          # one minibatch of images
    input   y : [B]                           # one minibatch of labels
    weights W                                 # trainable, an input not an embedded constant
    logits = forward(W, x)                    # inference-style ops only
    loss   = cross_entropy(logits, y)
    output  loss, grad(loss, W)               # autodiff appends the backward pass

program P = compile(G, target = H13)   # forward + backward in one program, one dispatch
M, V := 0                                     # adaptive optimizer moments, resident on the engine
for step in 1..300:
    grads = dispatch(P, x_batch, y_batch)     # forward and backward in one submitted graph
    M = beta1 * M + (1 - beta1) * grads               # update, on-device
    V = beta2 * V + (1 - beta2) * (grads * grads)
    W = W - lr_t * M / (sqrt(V) + eps)        # optimizer step writes resident W in place
    # M, V, W stay resident; host sends only (x_batch, y_batch, lr_t) next step
    # lr_t already absorbs loss scaling: it cancels in M / sqrt(V), do not divide out again
W_final = read(W)                             # read the weights back at a checkpoint
```

Listing: A resident-state training loop, keeping the weights and optimizer moments resident across all 300 steps. {#lst:c15-loop}

All three stages submit as one engine graph per step, so the weights and moments never cross the host boundary between steps.

## Reference: training correctness and the numeric edge

[Table](#tbl:c15-reference) collects the training correctness figures and the single guarded numeric edge this chapter establishes, from the gradient cosine through the final accuracies to the saturation threshold.

| Quantity | Value |
| --- | ---: |
| Registered-gradient cosine versus closed form | $1.0000$ |
| Final test accuracy, M1 | $0.9080$ after 300 steps |
| Final test accuracy, M5 | $0.9070$ |
| Cross-generation accuracy gap | one test sample in one thousand |
| Per-operation cross-generation fp16 difference | at most one unit in the last place |
| Loss-scale accuracies, M1 (128, 1024, 65536) | $0.9070$, $0.9080$, $0.9100$ |
| Conv weight-gradient saturation, width axis | 4094, where $65504 / 16 \approx 4094$ |
| First infinity at fixed shape | loss scale 512 (exact at loss scale 384) |
| Generations with the saturation | M1 and M2; non-saturating on A15 and later |

Table: The training correctness figures and the single guarded numeric edge. {#tbl:c15-reference}

## Capability and its scale

Training reaches the engine for the supported operation set, with the optimizer state resident across steps and the conv weight-gradient as the one guarded numeric edge.
The loop is dispatch-bound: the per-step work is small relative to the dispatch cost, so at this scale the engine is no faster than the host or the graphics processor.
