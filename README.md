# FinancialCalibrator.jl

[Documentation and examples](https://rcalxrc08.github.io/FinancialCalibrator.jl/)

[![CI](https://github.com/rcalxrc08/FinancialCalibrator.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rcalxrc08/FinancialCalibrator.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/rcalxrc08/FinancialCalibrator.jl/graph/badge.svg)](https://codecov.io/gh/rcalxrc08/FinancialCalibrator.jl)

Calibration of option pricing models (Carr-Madan Lévy processes, shifted lognormal mixture) to implied volatility surfaces.

```julia
using FinancialCalibrator, FinancialMonteCarlo
und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
```

See `examples/` for full calibration scripts (they need extra packages such as PyPlot / BenchmarkTools) and `data/json/` for the sample market data.

## Calibration

```julia
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT
und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)  # spot and dividends come from the underlying, not from the quotes

# the pricing method decides how each smile is priced
calibrate(model, CosMethod(1024), mkt)                  # any FinancialFFT method
calibrate(ShiftedLogNormalMixture(...), AnalyticMethod(), mkt)  # closed form
```

Spot and dividends are those of the model's `Underlying`. `calibrate(model, method, mkt, opt_method, opt_options; verbose)`
accepts any Optim optimizer and `Optim.Options`.

### GPU

With CUDA.jl loaded, strikes and volatilities can live on the device; the smiles are then priced there, with every
FinancialFFT method and `AnalyticMethod` (the execution mode of the method is moved to `CudaMode`):

```julia
using CUDA, Adapt
calibrate(model, CosMethod(1024), adapt(CuArray, mkt))
```

The execution mode is part of the method (FinancialMonteCarlo 0.6): `CosMethod(1024; mode = FinancialMonteCarlo.CudaMode())`
prices on the GPU with the market data on the host too.

