```{=latex}
\clearpage\pagenumbering{arabic}
```

# Introduction {.unnumbered}

The Apple Neural Engine (ANE) is among the most widely deployed machine-learning accelerators, and among the least documented.
Apple built the ANE into every Apple system on chip since the A11 in 2017 and the M1 in 2020, so nearly every iPhone and iPad active today has one.
The count of active Apple devices surpasses 2.5 billion (measured early 2026) [AppleActiveDevices2026].
On these devices it largely runs the vision, speech, and language models used by the operating system and its applications.
Of the programmable engines on an Apple chip (CPU, GPU, and ANE), the ANE is the most opaque: there is no public instruction set, no driver interface, and no documented way for a program to confirm that a computation ran on it.
An application reaches the ANE indirectly, through the lightly documented Apple Core ML framework, which treats the engine as one option behind a placement hint.
While Core ML is at least partly documented, what lies beneath it is not: the datapath and numerics, the performance and energy envelope, the compiler and program format, the kernel driver, the firmware, and the command protocol.

Here we report what was not previously available about the ANE, recovered by reverse engineering from the silicon datapath to the system interface.
Our evidence is of two kinds, and we check each against the other: we measure directly on Apple silicon, and we decompile the private runtime, compiler, kernel driver, and firmware statically while attaching to them at run time.
We find the engine reachable directly from an ordinary user process, below Core ML.
The ANE compiler lowers a graph to the engine's own program format, the runtime loads and dispatches it without the model framework, and the operations the compiler accepts need no special entitlement.
This direct route is what opens the rest of the device to measurement.
Some of the quantities we report, and some of the bytecodes we emit, are likely to break across operating-system updates.
Nothing here should be built into software that has to keep working.

We stage the performance results around a roofline.
On the M1 Max, the engine holds about 12 fp16 TFLOP/s of compute, bounded above by a DRAM-bandwidth ceiling.
The roofline has a ridge point near 141 FLOP per byte, a 2 MB working-set threshold, a 0.23 ms floor under any single dispatch, and efficiency near 0.37 picojoules per FLOP at peak compute.
On a 256-channel 3x3 convolution it runs about 3.8 times faster than the same chip's GPU and is 9 times more energy-efficient.
We take the power side of those figures from shipped device measurement tooling.

Reaching the engine is not the same as running an arbitrary computational graph on it.
The operations the engine executes are distinct from the ones a capability bit advertises.
We count a feature attested in the hardware tables or accepted by the compiler frontend only once a compile-and-run confirms it, and we find that several advertised operations, three-dimensional convolution among them, never lower to the engine at all.
Weight compression on the direct path cuts bandwidth, not only stored size.
On the unentitled engine, int4 lookup-table weights run about 2.37 times faster than fp16, and structured sparsity 1.55 to 1.64 times faster at 0.43 times the bytes.

Beneath the datapath lies the private stack the engine runs on: the compiler and its backend dialect, the on-disk program and container format, the kernel-driver IOKit ABI, the unencrypted firmware and its ninety-three-command host protocol, and the address-translation path that maps host buffers into the engine.
We decode twenty-eight compiler targets as well, spanning the A11 through A18 and M1 through M5 families, together with the rule that maps each M-series part to its internal H-series identity, the per-family operation floors, and an operation-by-device matrix.
Our cross-generation predictions held when we checked them on a second physical chip, and a seeded training run reproduced across generations to within 0.001 in final accuracy.

## Related work {.unnumbered}

The closest precedent is [Orion2026], concurrent work that characterizes and programs the engine for large-language-model training and inference.
A line of community reverse engineering precedes it: the tinygrad project recovered the HWX program format and the AppleH11ANEInterface IOKit path [tinygrad], Yoon's `ane` project built a reverse-engineered Linux driver and the `anecc` compiler [eilnANE], and Singh's recent series decodes the M4 engine [Singh2026], while Handley reconstructs the engine's hardware architecture from Apple's patent filings [Handley].

The `libane` native runtime exposes the engine to ordinary programs [libane], and whisper.cpp reaches it through the Core ML-routed path [whispercpp].
The long-maintained catalog of Hollemans [Hollemans] and Apple's own engineering note on deploying Transformers [AppleANETransformers] collect what is publicly known.
Several further repositories document direct access and runtime APIs [CommunityANE], and an earlier thesis treats decoupling on-device intelligence from the application on IoT hardware [Plyenkov2019].

The performance treatment is organized around the roofline model [Williams2009] and its later refinements: the energy roofline [Choi2013], the cache-aware roofline [Ilic2014], the instruction roofline [Ding2019], hierarchical roofline analysis [Yang2020], and its application to machine-learning accelerators [Verhelst2025].
It sits within a wider body of work that measures neural accelerators and edge inference: the in-datacenter analysis of the first tensor processing unit [Jouppi2017], smartphone deep-learning benchmarks [Ignatov2019], edge-platform inference benchmarking [Jayanth2024], energy-and-time roofline studies on edge accelerators [Prashanthi2025], NPU energy efficiency on microcontrollers [Fanariotis2025], LLM inference trade-offs across mobile NPU and GPU under sustained load [Tummalapalli2026], and on-device LLM roofline benchmarking [Bi2026].
The work closest in workload studies heterogeneous on-device LLM inference: fast NPU inference [Xu2025], GPU-NPU hybrid serving for long context [Moon2025], and mobile-SoC characterization for heterogeneous execution [Chen2025].
On Apple silicon specifically, Benazir and Lin study mixture-of-experts inference on the NPU [Benazir2026], Hübner and colleagues evaluate the M-series for HPC efficiency [Hubner2025], and the ML.ENERGY project [Zeus2025] treats the programmatic energy measurement the power figures here depend on.

We do not claim to be first to reach the engine.
The direct route we measure is released as the open-source ANEForge runtime [ANEForge2026], which this guide accompanies.
Our account differs from prior work in scope and in method, since the coverage runs the full stack, from the fp16 datapath down to the firmware and command protocol.
Beyond the access result we add a reachable-operation census, unentitled weight streaming, a validator-based prediction of an operation's reachability from a callable compiler validator, and an account of how the engine's compiler fuses a whole graph into a single program.
Earlier Apple-silicon rooflines are GPU-only; here we cover the engine, pair it with measured power, and give a batched-serving energy crossover.

## How to read this guide {.unnumbered}

We have written the two halves to be taken independently.
The front half, Parts I through V, covers using the engine: what the hardware is, how to reach it, how it performs, and how to fit real workloads to it.
The back half, Parts VI through IX, covers the architecture and system internals beneath that surface, down to the firmware and command protocol.
The appendices collect the operation-by-device matrix, decoded reference tables, glossary, and provenance record, and the references close the guide.

Every substantive claim carries an evidentiary mark.
A measured claim was observed directly on Apple silicon, primarily the M1 and M5.
Decompile-derived claims were read out of the disassembled runtime, compiler, kernel driver, or firmware, while predicted claims were inferred from a model or a per-chip table and are not yet confirmed on silicon.
Appendix E records the mark on every claim.
The methodology describes how we reached the engine, and the open questions name what we were unable to measure.
