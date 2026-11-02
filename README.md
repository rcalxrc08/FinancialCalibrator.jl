# FinancialCalibrator.jl

[![CI](https://github.com/rcalxrc08/FinancialCalibrator.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rcalxrc08/FinancialCalibrator.jl/actions/workflows/CI.yml)

Calibration of option pricing models (Carr-Madan Lévy processes, shifted lognormal mixture) to implied volatility surfaces.

```julia
using FinancialCalibrator, FinancialMonteCarlo
mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
```

See `examples/` for full calibration scripts (they need extra packages such as PyPlot / BenchmarkTools) and `data/json/` for the sample market data.
