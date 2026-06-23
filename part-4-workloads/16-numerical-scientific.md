# 16. Numerical and scientific computing

> Dense linear algebra and fixed-iteration numerics fit the engine, with the wide accumulator holding precision to a condition number of roughly $10^4$ before fp16 input rounding sets the error floor.
> Dense factorizations and full spectral decompositions run by unrolling into one static graph, reaching about $n = 32$ for the factorizations and about $n = 20$ for the spectral decompositions.
> The discrete Fourier transform runs as a matrix multiply, fp16-clean to about $N = 2048$.
> The single architectural limit is a runtime-sized output, not pivoting, since a data-dependent pivot is reached by cutting the graph at the pivot selection.

The Apple Neural Engine is a dense fp16 matrix engine with a wide accumulator.
That hardware structure decides which numerical and scientific kernels run on it without a precision penalty and which the architecture bars.
Dense linear algebra and fixed-iteration numerics fit; the dense factorizations and full spectral decompositions run by unrolling into one static graph; the discrete Fourier transform fits as a matrix multiply while the fast-transform butterfly fits only as a static unroll.
The one architectural limit is a runtime-sized output, not pivoting: a data-dependent pivot is reached by cutting the graph at the pivot selection.

## What the dense engine computes well

The three levels of dense linear algebra map onto the multiply array directly [AppleAccelerate].
A level-1 operation is a vector reduction or scaling: a dot product is a single contraction, and an axpy or a norm is a scaled add or a reduction.
A level-2 operation is a matrix-vector product, one matmul with a unit output dimension.
A level-3 operation is a matrix-matrix product, the native shape of the array.
The wide accumulator (chapter 3) holds the reduction in a register of fp32 class, so a representable sum comes back near exact and the only quantization is the fp16 rounding of the inputs and the output.

Iterative kernels inherit the same precision behavior because each step is a matmul or a reduction.
Power iteration for a dominant eigenpair, conjugate-gradient sweep for a symmetric positive-definite solve, and fixed-cycle generalized-minimal-residual solve for a general system all fold the matrix as a constant weight and run as one program.
Because the matrix is a fixed weight, the program size does not grow with $n$: a symmetric positive-definite solve runs unchanged from a small system up to $n = 512$ at the same relative error.
Across these solvers the wide accumulator holds the precision to a condition number of roughly $10^4$ before fp16 input rounding, not the accumulator, sets the error floor.

The measured envelope on the on-engine iterative solvers is a relative error near $10^{-3}$ at a condition number at or below $10^2$.
A dominant eigenpair from power iteration returns to about $10^{-15}$, limited by the spectral gap and not by the datapath, because the iteration is a repeated matmul whose fixed point is exact in the representable range.

The accumulator that holds the precision is measured, not assumed: a reduction accumulates in a register of fp32 class and rounds to fp16 only at the output port, with the silicon-probe worked examples and the radix-4 fan-in derived in chapter 3.
The consequence for these kernels is that a representable sum comes back near exact while a cancellation-heavy reduction has the input lanes round in radix-4 groups first, so the matrix-multiply contraction path is the more accurate route, exact to fp32 then rounded once at the output.
The compiler emits a precision warning steering a narrow reduction toward it.

## Factorizations and spectral decompositions by static unrolling

The direct factorizations of dense linear algebra run on the engine, and the assumption that they were barred is a misreading of the static-dataflow constraint.
No data-dependent control flow is not the same as no data-dependent computation.
A data-dependent value, a pivot or a selection, flows through a fixed graph when a comparison, a `select`, an `argmax`, and a matmul hold it, and a convergence-gated loop becomes a fixed sweep count chosen at compile time.
A factorization is thus expressible by unrolling its $O(n^3)$ recurrence into one static graph, and the closed-form recurrences that have no runtime branch unroll directly.

The explicit Gram-Schmidt QR, unpivoted Cholesky, and Doolittle LU run as one program and reach about $n = 32$ before the unrolled chain dominates compile time.
The full dense spectral decompositions run as well.
A fixed-sweep cyclic-Jacobi eigensolver has a compile-time-fixed sweep order and closed-form rotations with no pivot and no branch, and it converges in about six to ten sweeps for any symmetric input, so the full symmetric eigendecomposition unrolls and reaches about $n = 20$.
The generalized eigenproblem runs as a Cholesky factor followed by a triangular solve and that eigendecomposition.
The nonsymmetric eigenvalues run as an unshifted QR iteration assembled into one fused program, where the unshifted form avoids the data-dependent shifts and deflation that make the standard routine look unportable.
The full singular-value decomposition runs as the square root of the eigendecomposition of $A^\mathrm{T} A$, and a top-$k$ singular-value decomposition of a large matrix runs as a randomized sketch followed by the on-engine QR and small-block decomposition.

Pivoting is reached by cutting the graph at the pivot.
A pivoted LU runs with an on-engine `argmax` selecting the pivot row, expressed as a segmented graph: the `argmax` cuts the program into segments, and the data-dependent pivot index flows between them as a value rather than as a branch.
The measured relative error on the segmented pivoted LU is about $5 \times 10^{-4}$ at small $n$.

The single architectural limit is a runtime-sized output, not pivoting and not the recurrence.
A rank-revealing or adaptive-tolerance factorization emits only the values above a tolerance, and how many that is depends on the data, so a static program cannot emit a runtime-sized result and can only emit all $n$ values plus a mask.
That limit is independent of whether the arithmetic is fp16 or fp64.
What fp16 limits separately is the conditioning range, the roughly $10^4$ ceiling above; the Jacobi spectral decompositions hold to a condition number of about $10^1$ to $10^2$ because the squared system $A^\mathrm{T} A$ doubles the conditioning.
What graph size limits is $n$: the explicit factorizations materialize the matrix in-graph, so they are compile-size bound at the $n = 20$ to $32$ figures above, while the iterative solvers fold the matrix as a constant weight and run unchanged to $n = 512$.

## Stencils, integration, and series

A computation with a static iteration count fits, because the engine unrolls a fixed loop into one fused program and pays the per-dispatch floor once for the whole sweep.
A partial-differential-equation stencil iterated a fixed number of times is the clearest case.
A five-point stencil run thirty-two times as one fused graph keeps every intermediate resident in the working set and amortizes the dispatch floor across all thirty-two steps, which is why it is among the engine's widest efficiency margins against the GPU (chapter 10).
Explicit ordinary-differential-equation integration with a fixed step count, fixed-iteration Newton solve, and truncated power series each share this form: the trip count is known at compile time, so the whole iteration becomes one static graph.

A data-dependent trip count does not run.
A loop that runs until a residual falls below a tolerance, or until a step-size controller accepts a step, asks the engine to decide at runtime how many iterations to execute, the same data-dependent control flow that bars pivoting.
The fit is the fixed-budget form of each method: a convergence-gated loop becomes a fixed sweep count, chosen at compile time to cover the worst expected input.

## Fourier transform as a matrix multiply

The discrete Fourier transform of a length-$N$ signal is a linear map, so it is a matrix multiply against a fixed Fourier matrix.

$$X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N}, \qquad X = W x, \qquad W_{kn} = e^{-2\pi i k n / N}$$

The matrix $W$ is a compile-time constant, so the transform runs as a single matmul that folds $W$ as a weight, as [listing](#lst:c16-dft-matmul) gives in NumPy.

```python
import numpy as np

def dft_matrix(N):
    # W[k, n] = exp(-2j*pi*k*n/N): the compile-time-constant Fourier matrix.
    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)
    return np.exp(-2j * np.pi * k * n / N)

def dft(x):
    # The transform is one matmul X = W @ x; W folds as a fixed weight.
    # fp16-clean to about N = 2048; above that the fp16 rounding of W and x
    # dominates the wide-accumulator reduction.
    return dft_matrix(len(x)) @ x
```

Listing: The discrete Fourier transform expressed as one matrix multiply against a compile-time-constant Fourier matrix. {#lst:c16-dft-matmul}

The arithmetic is complex, and the engine computes in real fp16, so each complex value is held as a real and an imaginary pair and the complex product expands into four real multiplies and two real adds.
The wide accumulator holds the length-$N$ reduction, so the transform is fp16-clean to about $N = 2048$, above which the fp16 rounding of the matrix entries and the input begins to dominate the result.

The fast-transform factorization does not gain over the matmul form on this engine.
A fast Fourier transform replaces the $O(N^2)$ matmul with an $O(N \log N)$ chain of butterfly stages, but each stage is a small data shuffle and a complex multiply, and the shuffle is a static structure rather than a runtime loop.
The butterfly fits only when fully unrolled into a static graph, and at that point it is a deep chain of tiny operations that pays more dispatch and accumulates more fp16 rounding than the single dense matmul.
For the transform sizes the engine handles in range, use the matmul form.

## Sparse and pruned operands

A sparse matrix gives a speed gain, not only a storage gain, when it is a constant weight.
The engine has two independent sparsity mechanisms.
The first is a compute-time zero-skip: the multiply array detects a zero weight and skips the multiply, driven by a scan of the weight values, so it applies to any operand with zeros regardless of its storage format and pays off in a compute-bound layer.
The second is a sparse stream: the weight is stored as a one-bit keep-mask plus the packed fp16 nonzeros, fewer bytes cross DRAM, and the operand is reconstructed on chip, which pays off in a bandwidth-bound layer.

On the M1 the sparse stream is the measured gain for a pruned operand.
A convolution stack at about 63 percent zeros streamed as a mask plus nonzeros runs 1.55 to 1.64 times faster than the same weights stored dense.
The streamed weight file is at 0.43 times the dense size, and the effective bandwidth rises from about 29.5 to about 47.8 GB/s, the engine's weight-stream ceiling.
The reconstruction is lossless apart from the fp16 rounding of the kept values, so the result matches the dense computation to a cosine of $1.0000$.
The zero-skip mechanism, by contrast, returns about 1.01 times on the same bandwidth-bound stack, because skipping multiplies does not move a layer whose wall time is set by the DMA, and it needs a compute-bound layer to show.
[Table](#tbl:c16-sparsity) sets the zero-skip and sparse-stream mechanisms side by side: what each cuts, where it pays off, and its measured result on a pruned stack.

| Sparsity mechanism | What it cuts | Where it pays off | Measured M1 result |
| --- | --- | --- | --- |
| Compute-time zero-skip | multiply cycles | compute-bound layers | about 1.01x on a bandwidth-bound stack |
| Sparse weight stream | DRAM weight traffic | bandwidth-bound layers | 1.55 to 1.64x, weight file 0.43x dense |

Table: The two sparsity mechanisms on the M1 and where each one pays off, with the measured result on a pruned convolution stack at about 63 percent zeros. {#tbl:c16-sparsity}

## Discrete Fourier transform as one matmul

The Fourier matrix is embedded as a constant weight, and the transform runs as one matmul against the real and imaginary parts of the signal, which [listing](#lst:c16-dft-graph) builds as a compiled engine program.

```python
# The length-N discrete Fourier transform expressed as matrix multiplies.
# X[k] = sum over n of x[n] * exp(-2*pi*i * k*n / N).
# That sum IS a matrix-vector product: X = W x, where W[k,n] = exp(-2*pi*i * k*n / N).

# 1. Form the Fourier matrix as a compile-time constant (built once, on the host).
for k in 0 .. N-1:
    for n in 0 .. N-1:
        angle  = -2 * pi * k * n / N
        Wr[k,n] = cos(angle)               # real part of the Fourier matrix
        Wi[k,n] = sin(angle)               # imaginary part

# 2. Build the transform as matmuls. The signal is held as real and
#    imaginary parts, since the hardware works in real fp16.
build graph G:
    input xr : [N] fp16                    # real part of the signal
    input xi : [N] fp16                    # imaginary part of the signal

    # Complex product (Wr + i*Wi) * (xr + i*xi) split into real arithmetic:
    Xr = matmul(Wr, xr) - matmul(Wi, xi)   # real part of the transform
    Xi = matmul(Wr, xi) + matmul(Wi, xr)   # imaginary part of the transform
    # (Equivalently one matmul over stacked real-imag pairs; same arithmetic.)

    output Xr
    output Xi

# 3. Compile once: Wr and Wi fold in as fixed constant weights, so the whole
#    transform is a single program and the per-call floor is paid one time.
program = compile(G, target = H13)         # W is a constant weight, one program
output  = run(program, xr = real(signal), xi = imag(signal))

# The wide accumulator holds the length-N reduction, so the result stays
# fp16-clean to about N = 2048.
return output
```

Listing: The length-N discrete Fourier transform as a compiled engine program, with the Fourier matrices folded in as constant weights. {#lst:c16-dft-graph}

## Reference: what fits and what is architecture-limited

[Table](#tbl:c16-reference) collects the numerical kernel classes this chapter covers, marking each as fitting the engine or architecture-limited with the binding constraint on each.

| Kernel class | On the engine | Bound |
| --- | --- | --- |
| BLAS level 1, 2, 3 (dot, axpy, gemv, gemm) | fits | fp16 conditioning, roughly $10^4$ |
| Iterative solvers (power, conjugate gradient, generalized minimal residual) | fits | fp16 conditioning, roughly $10^2$ to $10^4$ |
| PDE stencil, explicit ODE, fixed-iteration Newton, truncated series | fits | static trip count required |
| DFT as $X = W x$ | fits | fp16-clean to about $N = 2048$ |
| Dense factorizations (QR, Cholesky, unpivoted and pivoted LU) | fits as a static unroll | graph size, about $n = 32$; pivot via a segmented `argmax` |
| Full spectral decompositions (symmetric, generalized, nonsymmetric eig, SVD) | fits as a static unroll | graph size, about $n = 20$; fp16 conditioning about $10^1$ to $10^2$ |
| Rank-revealing or adaptive-tolerance factorization | architecture-limited | data-dependent output size, no static-graph form |
| Data-dependent trip count | architecture-limited | no static-graph form |
| FFT butterfly | only as a static unroll | no gain over the dense matmul |

Table: Numerical kernel classes that fit the engine or are architecture-limited, with the binding constraint on each. {#tbl:c16-reference}
