# FinancialCalibrator.jl <img src="etc/logo.png" width="40">

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rcalxrc08.github.io/FinancialCalibrator.jl/)
[![codecov](https://codecov.io/gh/rcalxrc08/FinancialCalibrator.jl/graph/badge.svg)](https://codecov.io/gh/rcalxrc08/FinancialCalibrator.jl)
[![CI](https://github.com/rcalxrc08/FinancialCalibrator.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rcalxrc08/FinancialCalibrator.jl/actions/workflows/CI.yml)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Calibration of option pricing models (Lévy processes, Heston, shifted lognormal mixture, ...) to implied volatility
surfaces, with any pricing method of FinancialMonteCarlo's `pricer` interface, and differentiable: the calibrated
parameters are a function of the quotes whose derivatives come from the implicit function theorem.

```julia
using FinancialCalibrator, FinancialMonteCarlo
und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
```

See `examples/` for a calibration script (it plots with PyPlot, which is not a dependency) and `data/json/` for the sample market data.

## Calibration

```julia
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT
und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)  # spot and dividends come from the underlying, not from the quotes

# the pricing method decides how each smile is priced
calibrate(model, LewisMethod(100.0, 4001), mkt)         # any FinancialFFT method
calibrate(ShiftedProcess(LogNormalMixture(...), α), AnalyticMethod(), mkt)  # closed form
```

Spot and dividends are those of the model's `Underlying`. `calibrate(model, method, mkt, opt_method, opt_options;
parametrization, autodiff, polish, polish_backend, min_vega, verbose)` accepts any Optim optimizer and `Optim.Options`,
and polishes the result with Levenberg–Marquardt. The loss is half the sum of the squared implied volatility errors of
the out of the money quotes whose vega is at least `min_vega * S0`.

## Market data from the CBOE

```julia
id = CalibrationID("SPX", Dates.format(today(), "dduyyyy"), "USD")
und, mkt = adapter(retriever(id, CboeWebInputStrategy(rate = 0.04, dir = "market_data")))  # downloaded (and saved)
und, mkt = adapter(retriever(id, CboeInputStrategy(rate = 0.04, dir = "market_data")))     # the saved chain, offline
```

The forward and the dividends of every expiry are implied by put–call parity, and the out of the money implied
volatilities are resampled on a moneyness grid.

## Sensitivities to the quotes

```julia
using ForwardDiff
f(V) = calibrated_parameters(model, LewisMethod(100.0, 4001), mkt, V)   # the calibration, as a function of the quotes
ForwardDiff.jacobian(f, mkt.Volatility)                                  # ∂p*/∂v; Zygote and Mooncake work too
```

The sensitivity of any contract priced with the calibrated model, to every quote, is one gradient of the composition;
see the documentation.

### GPU

With CUDA.jl loaded, strikes and volatilities can live on the device; the smiles are then priced there, with every
FinancialFFT method and `AnalyticMethod` (the execution mode of the method is moved to `CudaMode`):

```julia
using CUDA, Adapt
calibrate(model, CosMethod(1024), adapt(CuArray, mkt))
```

The execution mode is part of the method (FinancialMonteCarlo 0.6): `CosMethod(1024; mode = FinancialMonteCarlo.CudaMode())`
prices on the GPU with the market data on the host too.


## Similar packages

- [Open Source Risk Engine (ORE)](https://github.com/OpenSourceRisk/Engine): C++ with Python bindings, built on
  QuantLib; exotics through a payoff scripting language and American Monte Carlo, sensitivities by bump and revalue
  or adjoint AD, XVA, and an OpenCL/CUDA compute environment for the simulations.
- [Derivatives Algorithms Library (DAL)](https://github.com/wegamekinglc/Derivatives-Algorithms-Lib): C++17 with
  Python and Excel bindings; adjoint AD built in, a scripting language for exotic payoffs, curve and short-rate
  calibration with adjoint Jacobians.
- [finmath-lib](https://github.com/finmath/finmath-lib): Java; Monte Carlo models (LIBOR market model, Bermudans by
  American Monte Carlo) with stochastic adjoint AD, conditional expectations included, and CUDA and OpenCL
  extensions for the simulations.
- [QuantLibAAD](https://github.com/auto-differentiation/QuantLib-Risks-Cpp) (formerly QuantLib-Risks): QuantLib
  compiled with the XAD adjoint AD tool, for the sensitivities of QuantLib's pricing engines in C++ and Python.
- [stochastic-rs](https://github.com/rust-dd/stochastic-rs): Rust with Python bindings; stochastic processes, pricers
  with Greeks, volatility surface calibrators, simulations on CUDA and Metal.
- [Hedgehog.jl](https://github.com/aleCombi/Hedgehog.jl): Julia, on SciML; Black–Scholes and Heston, European and
  American options, Greeks with ForwardDiff, implied volatility calibration.

FinancialCalibrator.jl differentiates through the calibration itself (implicit function theorem): the sensitivity of
any contract priced with the calibrated model to every quote is one gradient, with ForwardDiff, Zygote, Mooncake or
Enzyme, with the smiles priced on the CPU or on CUDA arrays.
