# 14. LLM case study

> Autoregressive decode is bandwidth-bound and dispatch-bound, the two regimes the engine loses, so it belongs on the GPU at every batch size.
> At a batch of 16 the GPU runs decode about 2.7 times faster and about 4.6 times more energy efficient than the engine.
> Int8 weights halve the weight traffic but leave the hybrid decoder at about 0.99 times the fp16 rate, because the step is dispatch-bound, not weight-bandwidth-bound.
> Send the prefill, encoder, and vision front end to the engine, and send the autoregressive decode to the GPU.

Autoregressive decode is the one major workload class where the engine is not the fastest.
The single-token decode step runs against the roofline of chapter 9, batching is the only serving control that moves throughput, and the engine's niche is on the other side of the model from the decoder.

## Why decode is not a compute problem

A decoder step generates one token.
It reads every weight in the model once, multiplies each against a single activation row, and produces one new row.
The arithmetic per byte of weight read is thus near one multiply-add per weight, which puts the step far below the 141 FLOP-per-byte ridge point of chapter 9.
Decode is bandwidth-bound: the step spends its time streaming weights from DRAM, not multiplying on the array.

A second cost adds on top of the bandwidth cost.
A single decoder layer is not one operation but a chain of projections, a normalization, attention block, and feed-forward pair, and on the direct path each is a separate dispatch.
A token through a transformer layer stack issues on the order of forty to fifty small dispatches, every one paying the per-eval floor of about 0.23 ms measured on the M1 in chapter 11.
Decode is thus dispatch-bound as well as bandwidth-bound, and both regimes are exactly the ones the engine loses, per the verdict table of chapter 11.

The wide accumulator does not help the decoder either.
A transformer down-projection loses precision in fp16 from the per-product rounding of its inputs under cancellation, the hazard named in chapter 3, not from the accumulator.
The decoder thus has an accuracy penalty on the engine on top of the speed penalty.

## Hybrid decoder placement

A decoder runs as a hybrid: the bandwidth-heavy, well-conditioned projections execute on the engine in fp16, and two precision-critical sites stay in wider precision off the engine.
The first is the residual stream.
Over twenty-two layers an fp16 residual add drops the small per-layer update, because adding a tiny delta to a large running sum loses the low bits.
The following normalization rescales but cannot recover them, thus the stream must accumulate in wider precision off the engine.
The second is the down-projection, the cancellation-heavy step of chapter 3, whose near-cancelled output over a contraction of 5632 takes about 3 percent error in fp16 on the engine, enough to flip the greedy argmax.

Per-operation placement follows [table](#tbl:c14-placement).
The query, key, gate, up, and output-embedding projections fit the engine in fp16; the value projection, output projection, and down-projection take a wider-precision path off the engine.

| Projection | End-to-end fp16 | Placement |
| --- | --- | --- |
| Query, key | Survives | Engine, fp16 |
| Gate, up | Survives | Engine, fp16 |
| Output embedding | Survives | Engine, fp16 |
| Value | Fails | Wider precision, off-engine |
| Output projection | Fails | Wider precision, off-engine |
| Down-projection | Fails | Wider precision, off-engine |

Table: Per-projection placement for a hybrid decoder: the engine holds the fp16 projections, three are held off-engine in wider precision. {#tbl:c14-placement}

Placement follows position, not per-operation error.
The gate and up projections survive on the engine despite per-operation error comparable to the value and output projections.
An operation survives on the engine when its result passes through a wider-precision step downstream.
The gate and up projections send the feed-forward delta through the off-engine down-projection, which insulates them.
The value and output projections send to the attention output and the residual directly, thus their error compounds through the key-value cache across the layer stack.

## Single stream against many streams

A single decode stream is serial by construction.
Token $t+1$ cannot start until token $t$ is produced, because it consumes that token, so there is nothing to overlap within one stream.
The host issues a dispatch, waits for the result, and issues the next, and the engine is idle between the dispatch floor and the next submission.

Concurrent independent streams behave differently.
Executing one stream releases the host thread, so several streams interleave their host-side work against each other's engine-side work and lift the aggregate token rate even though each individual stream stays serial.
This is a serving control for many requests, not a latency reduction for one.
[Table](#tbl:c14-serving-modes) compares the single-stream, multi-stream, and GPU-batched modes, what each improves, and how its throughput compares with the GPU.

| Serving mode | What it improves | Throughput against the GPU |
| --- | --- | --- |
| Single-stream decode | Latency for one request | Below the GPU; serial, no overlap |
| Multi-stream concurrent decode | Aggregate tokens per second | Higher aggregate than single stream, still below the GPU |
| GPU batched decode | Aggregate tokens per second | Fastest and most energy efficient at batch |

Table: Decode serving modes, what each improves, and how its throughput compares against the GPU. {#tbl:c14-serving-modes}

The multi-stream gain is real but bounded.
A fair batched comparison at a batch of 16 puts the GPU about 2.7 times faster and about 4.6 times more energy efficient than the engine on decode, because the engine path stays host-dominated while the GPU scales with batch.

## What int8 weights do and do not buy

Native int8 weights stream at half the bytes of fp16 and cut the weight-read traffic of the bandwidth-bound step.
That is a reduction in the primary cost of decode, and it is reachable on the direct path.

It does not make the engine faster on decode.
The int8 projections come back numerically correct, but the hybrid decoder runs at about 0.99 times the fp16 version, because once the model is small or the dispatch count is high the step is dispatch-bound and host-bound, not weight-bandwidth-bound.
Halving the weight traffic of a step whose wall time is set by forty-plus dispatch floors and host marshalling does not move the wall time.
An int8 gain needs a fused, engine-dominated decoder where the weight stream is the bottleneck, which the per-layer dispatch structure of a general decoder does not provide.
Batching, not quantization, is the control that moves serving throughput.

## Two serving controls and two hard caps

Two controls make the hybrid practical for serving.
Speculative decoding drafts several tokens with an inexpensive proposer and verifies all of them in one batched forward.
Because the weights are read once per forward regardless of the number of drafted tokens, the verification adds almost no cost.
On a TinyLlama decoder this lifts the rate from 35 tokens per second to as high as 128 on repetitive text and 42 to 56 on factual prose, with output identical to plain greedy decoding.
Batched prefill processes the prompt in chunks of several tokens at once instead of token by token, and because the engine matrix multiplies are flat in batch width, prefill latency collapses by about 2.3 to 5.9 times, bit-equal to serial prefill.

A multi-shape decoder also reaches two hard caps enforced by the device daemon.
The in-flight cap is 127 outstanding requests per program, set by a dispatch semaphore of 127 inside the in-memory model.
The loaded-program cap is near 128 programs per process: the next load fails with `GetANEFModel: must re-compile` and forces a recompile.
For multi-shape serving, where each sequence length or batch width is a distinct compiled program, the loaded-program cap bounds how many resident shapes a process can hold at once.

## Cache that stays resident

The one part of the decoder that fits the engine's execution model is the key-value cache.
The cache holds the keys and values of every prior token and grows by one row per step.
Re-streaming the whole cache through the host each token would add a copy proportional to the sequence length on top of the weight stream.

Instead, the cache stays resident on the engine across steps, as chapter 2 describes for resident state.
A program declares the cache as both an input and an output, and after compilation the runtime aliases the output buffer onto the input buffer, so the updated cache produced by one step is the cache consumed by the next without leaving the device.
A masked update writes the new key and value into the cache at the current position, with the position supplied as a small one-hot vector each step, as [listing](#lst:c14-cache-schematic) sketches.

```python
# Schematic of one resident-cache decode step.
# The cache output buffer is aliased onto the cache input buffer,
# so the cache never round-trips through the host.

compile(graph with inputs [x, slot_onehot, cache_in]
                outputs [logits, cache_out])

alias_output_to_input("cache_out" -> "cache_in")   # output buffer IS input buffer

# per token, the host sends only the new row and the slot, not the cache:
cache_out = cache_in * (1 - slot_onehot) + new_kv * slot_onehot
```

Listing: One resident-cache decode step, where aliasing the cache output buffer onto its input buffer keeps the key-value cache on the engine across steps. {#lst:c14-cache-schematic}

With the cache resident the host sends only the new token and the position one-hot each step, a few bytes, rather than the whole cache.
This removes the per-token cache copy and is the form in which a decoder is most efficiently expressed on the engine.
A resident cache removes one copy from a step whose wall time is still set by weight bandwidth and dispatch count, so it makes engine decode more efficient without making it as fast as the GPU.

## Two residency mechanisms

The engine has a native persistent-state type and a pair of read-state and write-state operations, the obvious mechanism for a resident cache.
On the unentitled runtime path that type parses but does not reach the engine.
The front end recognizes the state operations, but the backend rejects them during the conversion to the engine program, because the read-state has no state object to bind to: the runtime registers the cache operand as a plain input rather than as a persistent state, so the binding step fails before code generation.
The native route thus stays closed pending a compiler option that moves the state operations off the engine subgraph.

The buffer-aliasing route is open and verified.
It compiles a program whose cache is both an input and an output, then aliases the output buffer onto the input buffer after compilation, so the tensor persists on the engine across dispatches with no host re-supply.
The aliasing primitive is the same one that keeps optimizer state resident for on-engine training in chapter 15.
Two measurements confirm it on the M1.
A resident accumulator that adds one each step, aliased output onto input with no re-supply, returns 1, 2, 3, 4 over four dispatches, accumulating in place.
A resident cache of six slots, written by a masked positional update of $(t+1) \times 10$ at slot $t$, returns 10, 20, 30, 40, 50, 0 over five steps, each token written to its slot with the cache never leaving the device.
The coordination behind residency is a hardware event signal-and-wait primitive that sequences the producer and consumer of the cache buffer, emitted by the compiler rather than requested by the user.

## A resident-cache decode step

The procedure of [listing](#lst:c14-cache-step) compiles the decoder once with the cache as a paired input and output, aliases the buffers, then sends only the new token and its slot per step.

```c
/* Allocate one cache buffer object and bind it to BOTH the cache_in and cache_out ports, */
/* so the engine writes the updated cache back into the same resident buffer in place.    */
e5rt_buffer_object_alloc(&cache_buf, cache_nbytes, /*type=*/0);
e5rt_execution_stream_operation_retain_input_port(op, "cache_in", &cache_in_port);
e5rt_execution_stream_operation_retain_output_port(op, "cache_out", &cache_out_port);
e5rt_io_port_bind_buffer_object(cache_in_port, cache_buf);
e5rt_io_port_bind_buffer_object(cache_out_port, cache_buf);  /* alias: cache stays resident */

for (int t = 0; t < max_tokens; t++) {      /* one encode + execute per token */
    /* host marshals only the new row and slot; the cache is NOT re-sent */
    e5rt_execution_stream_operation_prepare_op_for_encode(op);
    e5rt_execution_stream_encode_operation(stream, op);
    e5rt_execution_stream_execute_sync(stream);
    e5rt_execution_stream_reset(stream);
}
```

Listing: The runtime calls for a resident-cache decode step, binding one cache buffer to both ports so the cache updates in place each token. {#lst:c14-cache-step}

## Reference: decode placement and the serving caps

[Table](#tbl:c14-reference) collects the decode placement figures and the two device-daemon serving caps this chapter measures.

| Quantity | Value |
| --- | ---: |
| Per-eval dispatch floor (M1) | 0.23 ms |
| Dispatches per transformer layer stack | 40 to 50 |
| Down-projection fp16 error over contraction 5632 | about 3 percent |
| Int8-hybrid decode rate versus fp16 | 0.99x |
| GPU versus engine decode speed at batch 16 | 2.7x faster |
| GPU versus engine decode energy at batch 16 | 4.6x more efficient |
| Speculative-decoding rate, TinyLlama | 35 to 128 tokens per second |
| Speculative-decoding rate, factual prose | 42 to 56 tokens per second |
| Batched-prefill latency reduction | 2.3x to 5.9x |
| In-flight cap per program | 127 outstanding requests |
| Loaded-program cap per process | near 128 (`GetANEFModel: must re-compile`) |

Table: The decode placement figures and the two device-daemon serving caps. {#tbl:c14-reference}

## Verdict: encoders, not decoders

The engine's niche is the other side of the same model.
Encoders and embedding models process a whole sequence in one forward pass of compute-bound matrix multiplies and normalizations, the engine's strong regime.
Chapter 11 measures a single-sentence encoder about 4.4 times faster than the GPU at low batch, with the crossover to the GPU only near a batch of 23.
Vision and convolution hold the same verdict more strongly.
Send the prefill, encoder, and vision front end to the engine, and send the autoregressive decode to the GPU.
