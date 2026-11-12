# Methods, backends and arrays

A calibration combines three choices that the package keeps independent: how a smile is priced, how derivatives are
taken, and where the arrays live. Each one is an argument, and changing one does not change the others.

## Pricing methods

The second argument of [`calibrate`](@ref) is any pricing method of FinancialMonteCarlo's `pricer` interface, so the
kind of calibration is the choice of that argument:

| Method | Package | Models |
|:--|:--|:--|
| `LewisMethod`, `CosMethod`, `CarrMadanLewisMethod`, `CarrMadanMethod` | [FinancialFFT.jl](https://github.com/rcalxrc08/FinancialFFT.jl) | every model with a characteristic exponent: Black–Scholes, Merton, Kou, Variance Gamma, NIG, Heston |
| [`AnalyticMethod`](@ref) | FinancialCalibrator.jl | closed forms: `ShiftedLogNormalMixture` |
| `MonteCarloConfiguration` | [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl) | every model that it simulates, with the same seed at every evaluation |
| `Vibrato`, `VibratoSaltando`, `PureLRM` | [VibratoMonteCarlo.jl](https://github.com/rcalxrc08/VibratoMonteCarlo.jl) | Monte Carlo with smooth estimators of the derivatives |

```julia
calibrate(model, CosMethod(2^10), mkt)                                   # Fourier
calibrate(model, LewisMethod(400.0, 40001), mkt)                         # Fourier, another quadrature
calibrate(slnm, AnalyticMethod(), mkt)                                   # closed form
calibrate(model, MonteCarloConfiguration(10^5, 1), mkt)                  # simulation
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

Mooncake takes the ChainRules rule through `Mooncake.@from_chainrules`. The test suite checks the Zygote gradient
against the ForwardDiff Jacobian (to 1e-10), also with a lazy cotangent (`sum`), and on the GPU.

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

On a surface of 15 quotes, a Zygote gradient takes 22 s with the defaults and 160 s with Mooncake for `x` (most of it
compilation), with the same result to 4e-16. Zygote for `x` does not work: it cannot differentiate the inner ForwardDiff
gradient of the conditions.

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

Measured on a laptop GPU (NVIDIA MX150) with the DAX surface of 108 quotes:

| | Host | Device | Agreement |
|:--|--:|--:|:--|
| `calibrate`, Black–Scholes, `CosMethod(1024)` | 12 s | 33–40 s | 4e-13 |
| ∂σ*/∂vᵢ, ForwardDiff, Black–Scholes | | 43–49 s | 2e-14 |
| `calibrate`, Variance Gamma, `LewisMethod(100.0, 4001)` | 9 s | 18 s | 4e-8 |
| ∂p*/∂vᵢ, ForwardDiff, Variance Gamma | | 70 s | 2e-7 |
| Zygote gradient, Black–Scholes | | | 4e-14 |
| Zygote gradient, Variance Gamma | | | 8e-8 |

A smile of 12 strikes is far too small to occupy a GPU, and these numbers time kernel launches: the device pays off with
many strikes per smile, or many quotes. The point here is that nothing in the calibration, nor in its derivatives,
needs to leave the device.
