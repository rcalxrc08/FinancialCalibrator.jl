```@raw html
<div class="fc-page-head">
<p class="fc-kicker"><span>Guide 02</span> Three independent choices</p>
```

# Methods, backends and arrays

```@raw html
<p class="fc-page-lede">A calibration combines three choices that the package keeps independent: <strong>how a smile
is priced, how derivatives are taken, and where the arrays live</strong>. Each one is an argument, and changing one
does not change the others.</p>
<div class="fc-meta"><span>FinancialFFT</span><span>closed forms</span><span>Monte Carlo</span><span>ForwardDiff · Zygote · Mooncake · Enzyme</span><span>CUDA</span></div>
</div>
<div class="fc-grid fc-grid-3">
<a class="fc-card fc-card-link" href="#Pricing-methods"><span class="fc-card-k is-blue">01 · Pricing method</span><h3>How a smile is priced</h3><p>Fourier for Lévy models and Heston, a closed form where one exists, Monte Carlo for everything else.</p><p class="fc-pick"><b>Default pick</b> <code>LewisMethod</code>; <code>CosMethod</code> where its series converges (Heston, Black–Scholes).</p></a>
<a class="fc-card fc-card-link" href="#Automatic-differentiation"><span class="fc-card-k">02 · AD backend</span><h3>How derivatives are taken</h3><p>Outside the calibration any backend; inside, four keywords, each an ADTypes backend.</p><p class="fc-pick"><b>Default pick</b> ForwardDiff everywhere; a reverse backend outside with many quotes.</p></a>
<a class="fc-card fc-card-link" href="#Arrays"><span class="fc-card-k is-amber">03 · Arrays</span><h3>Where the numbers live</h3><p><code>Float64</code> or <code>Float32</code>, on the host or on a CUDA device, derivatives included.</p><p class="fc-pick"><b>Default pick</b> the CPU for a surface of a hundred quotes; the GPU for large smiles.</p></a>
</div>
```

## Pricing methods

The second argument of [`calibrate`](@ref) is any pricing method of FinancialMonteCarlo's `pricer` interface, so the
kind of calibration is the choice of that argument:

| Method | Package | Models |
|:--|:--|:--|
| `LewisMethod`, `CosMethod`, `CarrMadanLewisMethod`, `CarrMadanMethod` | [FinancialFFT.jl](https://github.com/rcalxrc08/FinancialFFT.jl) | every model with a characteristic exponent: Black–Scholes, Merton, Kou, Variance Gamma, NIG, Heston |
| [`AnalyticMethod`](@ref) | FinancialCalibrator.jl | closed forms: `BlackScholesProcess`, `LogNormalMixture`, `ShiftedProcess` of either, see [Closed forms](closed_form.md) |
| `MonteCarloConfiguration` | [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl) | every model that it simulates, with the same seed at every evaluation |
| `Vibrato`, `VibratoSaltando`, `PureLRM` | [VibratoMonteCarlo.jl](https://github.com/rcalxrc08/VibratoMonteCarlo.jl) | Monte Carlo with smooth estimators of the derivatives |

```julia
calibrate(model, CosMethod(2^10), mkt)                                   # Fourier
calibrate(model, LewisMethod(400.0, 40001), mkt)                         # Fourier, another quadrature
calibrate(slnm, AnalyticMethod(), mkt)                                   # closed form
calibrate(model, MonteCarloConfiguration(10^5, 1), mkt)                  # simulation
```

The same Heston calibration of the DAX surface with three Fourier methods (measured by `docs/benchmarks.jl`):

```@example backends
Main.DocumentationFigures.engine_table()  # hide
```

A new model and method pair needs only a `pricer` method for a smile. The model is any FinancialMonteCarlo process;
the parameters the calibration moves are those of `FinancialMonteCarlo.get_parameters`, or the variables of a
[`Parametrization`](@ref).

The derivatives of the calibration (the Levenberg–Marquardt polish and [`calibrated_parameters`](@ref)) need a method
whose derivatives are those of its prices. FinancialFFT's methods differentiate their quadrature exactly; the
truncation range of `CosMethod` is held constant, which matters only where its series has not converged (Variance Gamma
and NIG at short maturities, see FinancialFFT.jl#64). A Monte Carlo method gives the pathwise derivative of its
estimator on its fixed paths.

## Automatic differentiation

### From outside

[`calibrated_parameters`](@ref) is a function of the quoted volatilities that every backend can differentiate, since
its rules come from the implicit function theorem and not from tracing the optimizer:

```julia
using ForwardDiff, Zygote
f(v) = calibrated_parameters(model, method, mkt, v)
ForwardDiff.jacobian(f, mkt.Volatility)                    # ∂p*/∂v, forward mode
Zygote.gradient(v -> w' * f(v), mkt.Volatility)            # wᵀ ∂p*/∂v, reverse mode (ChainRules)
```

Mooncake has a rule of its own, in the package extension loaded with Mooncake (ImplicitDifferentiation's ChainRules
rule needs a `RuleConfig`, which `Mooncake.@from_rrule` does not pass):

```julia
using Mooncake, ADTypes
import DifferentiationInterface as DI
DI.gradient(v -> w' * f(v), AutoMooncake(; config = nothing), mkt.Volatility)
```

### Support

Every derivative of the calibration runs through the implicit function theorem, so the outer backend never traces the
optimizer: it only meets the rule of the implicit function, and the inner backends meet the loss. The backends can be
mixed freely, forward over reverse or reverse over forward. The table shows the gradient of ``w^\top p^*(v)`` for
Variance Gamma with `LewisMethod(100.0, 4001)` on 15 quotes, relative to the ForwardDiff gradient on the host. Every
case ran in a fresh Julia process. The first call is timed after one calibration in that process, so it is the
compilation of the derivatives only; the device is a laptop GPU (NVIDIA MX150):

| Outer | Inner `x` / `y` | Host: first call | Host: next calls | Device: next calls | Error |
|:--|:--|--:|--:|--:|:--|
| ForwardDiff¹ | ForwardDiff / ForwardDiff | 14 s | 0.20 s | 0.32 s² | 0 (host), 5e-13 (device) |
| Zygote | ForwardDiff / ForwardDiff | 29 s | 0.15 s | 0.38 s | 2e-14, 2e-12 |
| Mooncake | ForwardDiff / ForwardDiff | 30 s | 0.48 s | 0.74 s | 2e-14, 2e-12 |
| Zygote | Mooncake / ForwardDiff | 138 s | 0.71 s | ✗³ | 2e-14 |
| Mooncake | Mooncake / ForwardDiff | 127 s | 0.41 s | ✗³ | 2e-14 |
| Enzyme, reverse⁴ | ForwardDiff / ForwardDiff | 87 s | 0.13 s | 0.43 s | 2e-14, 2e-12 |
| Enzyme, forward⁴ | ForwardDiff / ForwardDiff | 80 s | 0.12 s | ✗⁵ | 1e-15 |

One run per case: on this laptop the times of two runs differ by up to a third.

![Compilation and run time of the gradient per AD backend](assets/figures/bench-ad-light.svg)![Compilation and run time of the gradient per AD backend](assets/figures/bench-ad-dark.svg)

¹ With one chunk for all the quotes, `ForwardDiff.gradient(f, v, ForwardDiff.GradientConfig(f, v, ForwardDiff.Chunk{length(v)}()))`.
ForwardDiff's default chunk (8) splits 15 quotes in two, and each chunk runs the whole calibration again.
² One direction: ForwardDiff cannot seed a `CuArray`, so on the device it gives one derivative at a time
(`ForwardDiff.derivative` along a bump of one quote).
³ Mooncake differentiating the inner ForwardDiff gradient of the conditions meets a broadcast of dual numbers over a
`CuArray`, which it does not support.
⁴ Enzyme works through rules of the extension of FinancialCalibrator for EnzymeCore: they build the implicit function
outside of the differentiated code, then apply ImplicitDifferentiation's Enzyme rules to it as a constant (Enzyme
cannot trace the construction, and failed on the GPU when it did, #21). Those rules need the solver to return an
inferred type, which it does. On the device the first call compiles for 4 to 7 minutes here, 18 on Black–Scholes with
`CosMethod(1024)`.
⁵ Enzyme's forward mode seeds the gradient with one-hot arrays that it fills by scalar indexing, which a `CuArray`
refuses (EnzymeAD/Enzyme.jl#4048); the reverse mode works on the device.

What the numbers say:

- **Once compiled, a gradient costs about one calibration**, whatever the backend: here the solve takes 0.1 to 0.2 s
  (NelderMead, then the Levenberg–Marquardt polish), and the derivatives
  of the implicit function theorem about 8 ms. Reverse mode outside pays off with many quotes, not with these 15.
- **The first call is compilation**: Zygote and Mooncake compile their rules for the whole pricing method.
- **A reverse backend for `x` is reverse over forward**: Mooncake differentiates the ForwardDiff gradient of the loss,
  through the pricer. That compiles for minutes and runs 5 times slower than ForwardDiff for `x`; it can pay off only
  with many quotes, where forward mode needs one direction per quote.

The test suite checks ForwardDiff, Zygote, Mooncake and Enzyme (reverse, and forward on Black–Scholes) against the formula
``\partial p^*/\partial v = -H^{-1} \partial_v \nabla_p L``, and Zygote and Mooncake on the GPU.

### Inside

Four derivatives are taken inside, each by an [ADTypes.jl](https://github.com/SciML/ADTypes.jl) backend, through
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl):

| Keyword | Of | Default |
|:--|:--|:--|
| `autodiff` | the gradient of the loss, for a gradient based Optim optimizer (`LBFGS()`, `BFGS()`, ...) | `AutoForwardDiff()` |
| `polish_backend` | the Jacobian ``J`` of the volatility errors, in the Levenberg–Marquardt polish | `AutoForwardDiff()` |
| `conditions_backend` | the gradient ``g = \partial_x L``, the optimality conditions | `AutoForwardDiff()` |
| `implicit_backends = (x = ..., y = ...)` | the derivatives of ``g`` with respect to the quotes (`x`) and to the variables (`y`) | `AutoForwardDiff()` for both |

The last row differentiates the gradient of the row before it: a second order derivative, nested. ForwardDiff
everywhere suits a calibration, whose parameters are few: the Jacobian ``J`` has as many columns as parameters, and
the Hessian ``\partial_x g`` is small. The one place where reverse mode can pay is `x` with a reverse outer backend and
many quotes: the reverse rule needs ``(\partial_v g)^\top w``, which forward mode gets one quote at a time.

```julia
using ADTypes, Mooncake
calibrated_parameters(model, method, mkt, v; implicit_backends = (x = AutoMooncake(; config = nothing), y = AutoForwardDiff()))
```

The table above times these combinations. Zygote for `x` does not work: it cannot differentiate the inner ForwardDiff
gradient of the conditions.

### What each backend differentiates

The inner backends differentiate two functions: `FinancialCalibrator.loss` (for `autodiff`) and [`vol_errors`](@ref)
(for `polish_backend` and the conditions). `loss` returns ``\infty`` where the parameters cannot be priced, by catching
the exception. Zygote and Mooncake cannot differentiate a `catch`, so FinancialCalibrator gives them a rule for `loss`
(a ChainRules `rrule` for Zygote, a Mooncake rule in the extension) that runs the `try` itself around the reverse pass of
the loss without it, with a zero gradient where the loss is ``\infty``, as ForwardDiff and Enzyme give
(FinancialCalibrator.jl#40). Gradients on the DAX surface, against ForwardDiff
(shifted lognormal mixture with `AnalyticMethod`, Variance Gamma with `LewisMethod(100.0, 2001)`):

| Backend | `vol_errors` | `loss` |
|:--|:--|:--|
| ForwardDiff | ✓ | ✓ |
| Zygote | ✓ | ✓ (a rule, FinancialCalibrator.jl#40) |
| Mooncake | ✓ | ✓ (a rule, FinancialCalibrator.jl#40) |
| Enzyme, reverse | ✓ | ✓ |

Enzyme needs no runtime activity with FinancialMonteCarlo 0.9 or later, whose curves are `isbits`
(FinancialMonteCarlo.jl#79); with 0.8, a model rebuilt from active parameters on the dividend curve that
[`adapter`](@ref) returns needs `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`. When calling
`Enzyme.autodiff` directly, pass the model, the market data and the cache of `loss_cache` as `Const` arguments: captured in
a closure annotated `Const`, they make Enzyme ask for runtime activity on any underlying. Every backend works as the
`autodiff` of a gradient optimizer such as `LBFGS()` on the mixture and on Variance Gamma, whose line search on DAX
reaches parameters that cannot be priced. On Heston with `CosMethod`, Enzyme does not (`EnzymeNoDerivativeError`,
FinancialCalibrator.jl#41).

Zygote is correct but slow here: a gradient of `vol_errors` costs 1.6 ms on the mixture (ForwardDiff 0.11 ms) and 16 ms
with Lewis (ForwardDiff 2.7 ms), with FinancialMonteCarlo 0.9 (laptop, i7-8550U). With 0.8 the mixture took 7.7 ms, most
of it Zygote tracing the model setup (rebuilding the model and integrating the dividend curve element by element):
FinancialMonteCarlo 0.9 gives `set_parameters!` and the `integral` of a curve reverse rules (FinancialMonteCarlo.jl#80).
The
inner backends have few parameters to differentiate, which is what ForwardDiff is for; Zygote belongs outside, on
`calibrated_parameters`, where it meets only the rule of the implicit function.

Every backend differentiates the pricing code itself, which is written for it: no branches, validations outside the
derivatives, and for the closed forms a kernel whose Zygote gradient is inferred (see
[Other closed forms](closed_form.md#Other-closed-forms)).

### Which backend

`bench/` times every combination of backends, each in a fresh Julia process, on three models with 99 quotes of the DAX
surface (synthetic volatilities of a known model, so the optimum is well conditioned): the shifted lognormal mixture
with [`AnalyticMethod`](@ref), Variance Gamma with `LewisMethod(100.0, 4001)` and Heston with `CosMethod(1024)`. The
times are run times once compiled (laptop: i7-8550U, NVIDIA MX150; Enzyme 0.13.221, CUDA 6.5.1, 2026-11-20); the
first call compiles for 1 to 3 minutes with Enzyme, Zygote and Mooncake, against 10 ms with ForwardDiff. ✗ marks a
combination that fails; FinancialCalibrator.jl#41 explains each one.

**Inside: ForwardDiff or Enzyme forward.** The derivatives in the parameters (host, `Float64`):

| Backend | `loss` gradient: mixture | VG | Heston | `vol_errors` Jacobian: mixture | VG | Heston |
|:--|--:|--:|--:|--:|--:|--:|
| ForwardDiff | 97 μs | 6.5 ms | 9.0 ms | 90 μs | 6.0 ms | 8.5 ms |
| Enzyme, forward | 85 μs | 4.3 ms | 7.5 ms | 89 μs | 4.6 ms | 6.9 ms |
| Enzyme, reverse | 128 μs | 9.2 ms | 10 ms | 1.5 ms | 0.74 s | 0.54 s |
| Zygote | 1.5 ms | 20 ms | 26 ms | 72 ms | 1.1 s | 1.0 s |
| Mooncake | 0.40 ms | 69 ms | 58 ms | 12 ms | 1.8 s | 2.8 s |

A gradient in 3 to 5 parameters is what forward mode is for. Reverse mode builds the Jacobian of `vol_errors` one quote
at a time, 100 to 300 times slower. In `calibrate` (`LBFGS()`):

| `autodiff` / `polish_backend` | Mixture | VG | Heston |
|:--|--:|--:|--:|
| ForwardDiff / ForwardDiff (the default) | 4.6 ms | 0.34 s | 0.43 s |
| Enzyme forward / ForwardDiff | 4.8 ms | 0.21 s | ✗ |
| Enzyme forward / Enzyme forward | 5.0 ms | 0.26 s | ✗ |
| ForwardDiff / Enzyme forward | 5.1 ms | 0.24 s | 0.38 s |
| Enzyme reverse / ForwardDiff | 6.2 ms | 0.32 s | ✗ |
| Zygote / ForwardDiff | 0.16 s | 0.53 s | 0.61 s |
| Mooncake / ForwardDiff | 33 ms | 2.7 s | 3.0 s |
| ForwardDiff / Enzyme reverse, Zygote, Mooncake | 19 ms to 0.81 s | 7.6 to 20 s | 4.8 to 13 s |
| `NelderMead()`, ForwardDiff polish | 6.9 ms | 0.71 s | 0.68 s |

Enzyme forward saves up to a third on the Fourier methods, but the default stays ForwardDiff: it compiles in
milliseconds and works on every model.

**Outside: any reverse backend, or ForwardDiff.** The gradient of ``w^\top p^*(v)`` with the default inner backends,
and the full Jacobian ``\partial p^*/\partial v``:

| Outer backend | Gradient: mixture | VG | Heston | Jacobian: mixture | VG | Heston |
|:--|--:|--:|--:|--:|--:|--:|
| ForwardDiff | 26 ms | 1.2 s | 1.4 s | 44 ms | 1.4 s | 1.4 s |
| Zygote | 20 ms | 1.5 s | 1.4 s | 56 ms | 2.6 s | 5.6 s |
| Enzyme, reverse | 18 ms | 1.5 s | 1.6 s | 68 ms | 3.4 s | 7.4 s |
| Enzyme, forward | 59 ms | 4.1 s | 5.2 s | 82 ms | 4.3 s | 5.9 s |
| Mooncake | 44 ms | 2.2 s | 4.7 s | 113 ms | 5.4 s | 11 s |
| Zygote, inner `x` Mooncake | 10 ms | 1.6 s | 2.4 s⁵ | | | |

⁵ With FinancialFFT ≥ 0.6.3 (its Mooncake overlay of the complex square root covers the dual numbers of the inner
ForwardDiff gradient, #41); measured 2026-11-21 on the same laptop, outer Mooncake 3.3 s.

The outer backend meets only the rule of the implicit function, so the inner derivatives set the cost, and the reverse
backends tie with ForwardDiff on 99 quotes. Enzyme as an inner backend of [`calibrated_parameters`](@ref) fails in
this benchmark (#41). For the full Jacobian, ForwardDiff.

**Arrays: the device from about a thousand quotes, and `Float32` only there.** Heston with `CosMethod(1024)` on 99 DAX
quotes and on a large surface (4 maturities × 500 strikes, 1929 quotes in the loss):

| | Host `Float64` | Host `Float32` | Device `Float64` | Device `Float32` |
|:--|--:|--:|--:|--:|
| `loss` gradient, ForwardDiff, 99 quotes | 9.5 ms | 11 ms | 9.4 ms | 6.8 ms |
| `calibrate`, `LBFGS()`, 99 quotes | 0.33 s | 0.96 s | 0.42 s | 1.0 s |
| Gradient of ``w^\top p^*``, Zygote, 99 quotes | 1.3 s | 2.0 s | 3.2 s | 2.5 s |
| `loss` gradient, ForwardDiff, 1929 quotes | 25 ms | 64 ms | 15 ms | 8.6 ms |
| `calibrate`, `NelderMead()`, 1929 quotes | 1.4 s | 5.9 s | 1.3 s | 0.75 s |
| Gradient of ``w^\top p^*``, Zygote, 1929 quotes | 57 s | 157 s | 40 s | 21 s |

Mooncake differentiates the Fourier methods on the device too (Variance Gamma and Heston, the `loss` gradient matches
ForwardDiff to 4e-15), through a rule of the extension for the concatenation of the maturities
(chalk-lab/Mooncake.jl#1484), and the closed forms (Black–Scholes and the mixtures, 1e-14): their device broadcasts
refuse a callable with state, so a rule of the CUDA and Mooncake extension differentiates the same prices with the
numbers of the kernel passed as broadcast arguments (#41 group 3c). The Mooncake Jacobian of `vol_errors` on the device
fails in Mooncake itself, which refuses a `CuArray` output (#41 group 3d, chalk-lab/Mooncake.jl#1485).

On the host, `Float32` is slower than `Float64` and loses digits: the parameters agree to 1e-4 and their derivatives
to 1e-3. On the device `Float32` halves the times of the large surface. On 1929 quotes the gradient of
``w^\top p^*`` is dominated by the derivative of the conditions in the quotes (`implicit_backends.x`, forward mode:
one direction per quote); a reverse backend for `x` needs one pullback per parameter instead, see #41. Its full Jacobian with Zygote takes 5.5 minutes on the host and did not end in 50 minutes
in host `Float32`.

## Arrays

### Floating point types

[`MarketData`](@ref) and [`CalibratedData`](@ref) are parametric in the floating point type. `adapter(raw, Float32)`
reads a surface in `Float32`, and the losses are computed in it.

### GPU

The strikes and the volatilities of [`MarketData`](@ref) can live on the GPU. With CUDA.jl loaded, every smile is then
priced on the device (the extension moves the method to `CudaMode`), the implied volatilities are a broadcast on the
device, and so are the derivatives: the dual numbers of ForwardDiff and the cotangents of Zygote are `CuArray`s too.
The small linear systems (the Levenberg–Marquardt step, the implicit function theorem) come back to the CPU.

```julia
using CUDA, Adapt
gmkt = adapt(CuArray, mkt)                                            # strikes and volatilities on the device
calibrate(model, CosMethod(1024), gmkt)
calibrated_parameters(model, CosMethod(1024), gmkt)
Zygote.gradient(V -> sum(calibrated_parameters(model, CosMethod(1024), gmkt, V)), gmkt.Volatility)[1]   # a CuMatrix
```

Measured on a laptop GPU (NVIDIA MX150) with the DAX surface of 108 quotes, once compiled:

| | Host | Device | Agreement |
|:--|--:|--:|:--|
| `calibrate`, Black–Scholes, `CosMethod(1024)` | 0.06 s | 0.41 s | 4e-15 |
| `calibrate`, Variance Gamma, `LewisMethod(100.0, 4001)` | 0.36 s | 0.71 s | 3e-8 |
| one loss, Black–Scholes, `CosMethod(1024)` | 1.1 ms | 8.5 ms | |
| one loss, Variance Gamma, `LewisMethod(100.0, 4001)` | 2.7 ms | 7.2 ms | |
| Zygote gradient, Black–Scholes | | | 4e-14 |
| Zygote gradient, Variance Gamma | | | 8e-8 |

How the cost of one loss grows with the strikes of a smile, on the same laptop (Black–Scholes, `CosMethod(4096)`,
measured by `docs/benchmarks.jl`):

![Loss time against the number of strikes on CPU and GPU](assets/figures/bench-scaling-light.svg)![Loss time against the number of strikes on CPU and GPU](assets/figures/bench-scaling-dark.svg)

A smile of 12 strikes is far too small to occupy a GPU, and these numbers time kernel launches: a loss costs 7 to 9 ms
on the device whatever the model and the method. The device pays off with many strikes per smile, or many quotes. The
point here is that nothing in the calibration, nor in its derivatives, needs to leave the device.

#### Float32

Consumer GPUs compute in `Float64` at a fraction of their `Float32` rate (1/32 on the MX150), and `adapter(raw, Float32)`
gives a surface in `Float32`. On the DAX surface this changes nothing on the device, since the launches dominate: a loss
costs 9.1 ms (Black–Scholes, Cos) and 7.6 ms (Variance Gamma, Lewis), and a calibration takes about as long (Black–Scholes
0.46 s, Variance Gamma 0.75 s, parameters within 2e-4 of the host in `Float32`). On a smile large
enough to occupy the device, `Float32` pays: 2000 strikes with `CosMethod(4096)`:

| | Host | Device |
|:--|--:|--:|
| `Float64` | 17.7 ms | 14.3 ms |
| `Float32` | 17.6 ms | 6.7 ms |
