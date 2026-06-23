# 2. Execution model

> The engine runs as an autonomous coprocessor: a network compiles once into the engine's program format, then dispatches many times against that one program.
> The compile phase is costly and belongs out of the hot loop; the dispatch phase binds operands, posts one mailbox command, and waits.
> The compiled program is a static graph the hardware walks, so control flow is fixed at compile time and cannot depend on runtime values.
> A buffer can stay resident across dispatches, so a key-value cache or optimizer state persists in place without a host round-trip.

## Compile once, dispatch many

Work reaches the engine in two phases whose costs are far apart.
The public surface exposes only a load-and-predict view of this split [AppleCoreML]; the two phases below are the mechanism beneath it.
The compile phase turns a network into the engine's program format: the compiler lowers the operation graph, lays out weights for the streaming datapath, and produces a loadable program.
The dispatch phase runs that program against a set of operands and reads back the output.

The compile phase is costly.
It runs the full lowering and layout pipeline, writes a program to a content-addressed cache on disk, and the first dispatch of a freshly compiled program pays a further one-time cost to produce the loadable hardware form.
The dispatch phase binds operand buffers, hands the program to the engine, and waits for completion.

[Table](#tbl:exec-phases) sets the two phases against the runtime call surface, with when each runs, its cost, and the calls in each.

| Phase | When it runs | Cost | Example calls |
| --- | --- | --- | --- |
| Compile | once, ahead of the loop | full lowering and layout, written to disk | compile the network, open the program library, retain the program function, load the function for execution |
| Dispatch | once per frame, token, or request | bind operands, post command, wait | encode the operation, execute the stream, read the output |

Table: The compile and dispatch phases, when each runs, its cost, and the runtime calls in each. {#tbl:exec-phases}

The runtime exposes the two phases as distinct call families: [Listing](#lst:exec-compile) gives the compile-side calls, which run once, and [Listing](#lst:exec-dispatch) the dispatch-side calls, which run on every request.

```c
/* once, out of the hot loop */
e5rt_e5_compiler_compile(compiler, model_path, options, &library);
e5rt_program_library_retain_program_function(library, name, &function);
e5rt_program_function_load_for_execution(function);
```

Listing: The one-time compile-side calls that turn a network into a loaded program. {#lst:exec-compile}

```c
/* once per frame, token, or request */
e5rt_execution_stream_encode_operation(stream, op);
e5rt_execution_stream_execute_sync(stream);
```

Listing: The per-call dispatch-side calls that encode and run the loaded program. {#lst:exec-dispatch}

Chapter 6 gives the full create-bind-encode-execute sequence, including the compute operation and the input and output port binding.

The compile phase does not produce one program form but three, lowered at successive layers, which [table](#tbl:exec-forms) names with what each is and where it lives.

| Representation | What it is | Where it is |
| --- | --- | --- |
| the bundle | a flat-buffer container whose fused graph collapses to a three-op chain, a cast in, the engine inference, a cast out, with a parametric per-op descriptor whose size does not grow with the tensor | the on-disk content-addressed cache |
| the program image | a signed executable the kernel loader parses, magic `0xbeefface`, with a register-write text section, a weight section, constants, and scratch | materialized below the host boundary at load |
| the firmware container | the firmware's load format, a three-level container keyed by program identity, with generic, kernel, text, operation, and procedure sections | resident on the engine |

Table: The three program representations a compiled network passes through, from the cached bundle to the firmware load format. {#tbl:exec-forms}

The loader expands the parametric descriptor in the bundle into the explicit register-write program below the host boundary, so the shape-specific program appears in no host buffer.
This is why the compile cost is paid once on disk and a further one-time cost on the first dispatch, when the loadable hardware form is produced.

## Host drives an autonomous coprocessor

The engine is on the system on chip with its own controller and its own local memory.
The host never reads or writes the engine's compute registers and never steps it instruction by instruction.
It hands over a compiled program and the operand buffers, signals the engine through a command mailbox, and waits for a completion notification.

The mailbox is a ring buffer of command records shared between the host and the engine's controller.
To start work, the host writes a command that names the program and its operands and rings a doorbell.
The controller picks the command up, drives the datapath through the work, and signals completion back across the same channel.
Operand buffers are mapped through the engine's own address translation unit, so the engine reads inputs and writes outputs directly in memory the host prepared.
The host waits on the completion signal; it does not supervise the computation.

Once the command is posted, the host CPU is idle with respect to that work and can prepare the next operands or post more commands.
Several programs from several processes can have work outstanding at once; the engine time-shares itself across them without host involvement.

Each inference is a procedure call that contains its operand buffers, its optional wait, signal, and shared events, and a set of task-descriptor partitions.
The firmware pushes one engine request per partition onto a task queue, and the engine preempts at task-queue granularity with a mid-flight abort, so a higher-priority program does not wait for a lower-priority one to drain.
The host posts a command through a header that names the command, its size, a priority in the range 0 to 7, and the program, process, and procedure identities.
A secure mode can claim the engine exclusively by quiescing and power-cycling around the boundary, which is how protected-content work is isolated.

A single dispatch stream keeps one operation in flight reliably, but overlapping two or more streams in one process is the unfinished path on the M1.
The completion event for the first stream signals and its waiter returns, while the completion notification for a second concurrently overlapped stream does not fire, so its waiter blocks.
The runtime has the controls that would change this, a low-latency event path and a submit call with a timeout, but the default serialized path is the sound one.
A caller that needs aggregate throughput runs independent streams rather than overlapping them in one, since sequential decode cannot overlap with itself in any case.

## What one dispatch costs

A single dispatch is governed by the cost of getting to the engine and back, not by the engine compute itself.
Measured live on the M1 with a read-only trace, a tiny graph of a 3-by-3 convolution from 8 channels to 8 with padding 1, then a relu, then a mean, runs in a hot loop of about 2000 iterations.
Each call costs about 190 microseconds of wall-clock time.
About 98 percent of that is software and firmware dispatch overhead rather than engine compute.

[Table](#tbl:exec-budget) breaks the per-call budget into its stages, with the cost of each from the user-space binding through the firmware round trip to kernel-side completion.

| Stage | Cost |
| --- | --- |
| User-space binding, runtime, and host fp16 input and output copy | about 25 microseconds |
| Building the firmware request | about 16 microseconds |
| Firmware kick: the doorbell, which returns asynchronously | about 2 to 3 microseconds |
| Firmware round trip | about 130 microseconds |
| Kernel-side completion processing | about 10 microseconds |

Table: The per-call latency budget of a single small dispatch on the M1, from the user-space binding through the firmware round trip to kernel-side completion. {#tbl:exec-budget}

The kernel user-client submit, the `ANE_ProgramSendRequest` external method, takes about 163 microseconds from entry to return.
Completion is interrupt-driven: an interrupt handler fires about twice per inference, and the asynchronous-message completion path is not used on this synchronous small-model path.
During that firmware round trip the firmware wakes, picks the command off its queue, executes, and signals back.
It is not sub-splittable from user space with read-only tools, because the firmware per-task-descriptor latency profiler is gated.

## Submissions serialize at one in flight

The driver keeps at most one firmware command in flight at a time, a single-pending-queue scheduler.
Two concurrent submission threads thus serialize, measured at 1.04 times, so the round trip is not hidden by overlapping requests.

## A walked graph, not a decoded stream

The compiled program is a static graph of work segments that the hardware walks, not a stream of instructions that a processor decodes.
There is no program counter to read and no microcode to dump.
The compiler fixes the order and shape of every segment ahead of time, and the engine's controller advances through that fixed structure, programming the data movement engines and the multiply array for each segment in turn.

A direct consequence is that control flow must be static.
The graph has a shape decided at compile time, so the work the engine performs cannot depend on values computed during the run.
Data-dependent branching does not execute on the engine: no path through the walked graph selects itself from a runtime value.
A network that needs such a branch must resolve it on the host or restructure it so the branch becomes a fixed computation, for example a mask applied to both sides rather than a choice between them.
A loop with a fixed trip count unrolls into a fixed graph and is admissible; a loop whose length depends on the data is not.

The same property explains the absence of a readable instruction trace.
The unit the engine executes is a precompiled segment of data movement and multiply work, parameterized by operand addresses and shapes.
The fine-grained register program that drives the silicon is materialized below the host boundary at load time and never appears in a host buffer.
The program format is the subject of a later chapter.

## State kept resident across dispatches

A dispatch does not have to round-trip every tensor through the host.
The engine can keep a buffer resident in its working set across calls, so a value produced by one dispatch is available to the next without a copy back to the host and a copy forward again.
The mechanism aliases an output buffer of one call to an input buffer of the following call, so the data persists in place between dispatches.

The aliasing reuses the same port-binding calls that chapter 6 uses for ordinary I/O.
One buffer object is bound to the output port of the operation and to the input port of the next dispatch, so the dispatch that writes the held tensor and the dispatch that reads it name the same memory, as [listing](#lst:exec-resident) shows call by call.

```c
/* One buffer object holds the resident state (a KV-cache or optimizer state). */
e5rt_buffer_object_alloc(&state_buf, nbytes, /*type=*/0);

/* Bind it to BOTH the output port that writes the new state ... */
e5rt_execution_stream_operation_retain_output_port(op, "state_out", &out_port);
e5rt_io_port_bind_buffer_object(out_port, state_buf);

/* ... and the input port that reads it on the next step: same buffer object. */
e5rt_execution_stream_operation_retain_input_port(op, "state_in", &in_port);
e5rt_io_port_bind_buffer_object(in_port, state_buf);

for (int step = 0; step < n; step++) {
    /* Send only the small per-step input (a token or a minibatch). */
    /* schematic: write the step input into its own bound port, not state_buf */
    e5rt_execution_stream_encode_operation(stream, op);
    e5rt_execution_stream_execute_sync(stream);
    /* state_buf now holds the updated state; it is never re-sent from the host. */
    e5rt_execution_stream_reset(stream);
}
```

Listing: Keeping a state buffer resident across dispatches by binding it to both an output port and the next step's input port. {#lst:exec-resident}

Any multi-step computation that holds state uses this mechanism.
An autoregressive decoder keeps its key and value cache resident, so each step appends the new entry in place rather than restreaming the whole cache through the host every token.
A training loop keeps its optimizer state resident across steps for the same reason.
The host sends only the small per-step inputs, a new token or a minibatch, and reads the resident buffers back at a checkpoint.
The large held tensor stays on the engine instead of crossing the host boundary twice per step.

The output-to-input aliasing is the reachable face of the firmware's data-chaining subsystem, which keeps an output set resident and chains it as the next call's input.
This is the route on the M1, rather than the engine's native persistent-state operations.
The M1 task descriptor has no in-place resident-state data-movement engine: the encoders for that path are stubbed with the message that the data-movement form is not supported on this architecture, and the engine's native state operations are rejected when the program compiles.
The held tensor is thus written as one call's output and read as the next call's input through one bound buffer, with the positional write done as a standard masked update against a small position vector sent each step.
A resident accumulator built this way returns 1, 2, 3, 4 over four dispatches with no host re-send, and a resident cache fills each slot in place across steps, both confirmed on the M1.

## Compile out of the loop, dispatch inside it

The cost split dictates the loop structure: compile once before the loop, then dispatch the loaded program against fresh operands on every iteration.
A content hash keys the compiled program, so recompiling an unchanged network is a cache hit rather than a second lowering pass.

[Listing](#lst:exec-loop) compiles and loads once, then dispatches the loaded program per frame, per token, or per request.

```c
/* Once, out of the hot loop: compile and load. A cache hit skips the lowering. */
e5rt_e5_compiler_compile(compiler, model_path, options, &library);
e5rt_program_library_retain_program_function(library, fn_name, &function);
e5rt_program_function_load_for_execution(function);
e5rt_precompiled_compute_op_create_options_create_with_program_function(&op_opts, function);
e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options(&op, op_opts);
e5rt_execution_stream_operation_retain_input_port(op, "x", &in_port);
e5rt_io_port_bind_buffer_object(in_port, in_buf);   /* bound once, refilled per frame */
e5rt_execution_stream_create(&stream);

for (int frame = 0; frame < n_frames; frame++) {
    /* Inside the loop: write the next frame, then encode, execute, reset. */
    e5rt_execution_stream_operation_prepare_op_for_encode(op);
    e5rt_execution_stream_encode_operation(stream, op);
    e5rt_execution_stream_execute_sync(stream);
    e5rt_execution_stream_reset(stream);
}
```

Listing: The compile-once, dispatch-many loop, with compile and load above the loop and bind and dispatch inside it. {#lst:exec-loop}
