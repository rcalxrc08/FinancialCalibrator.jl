```@raw html
<img class="pkg-hero-logo" src="assets/logo.png" alt="FinancialCalibrator.jl logo" width="96" height="96">
```

# FinancialCalibrator.jl

```@raw html
<p class="pkg-roles"><b>Calibration</b> · <b>Implicit differentiation</b> · <b>Any pricing method</b> · <b>GPU</b></p>
<div class="pkg-intro">
<p>FinancialCalibrator.jl fits option pricing models to an implied volatility surface. A model is any process of
FinancialMonteCarlo.jl, a smile is priced by any method of its <code>pricer</code> interface (Fourier, closed form,
Monte Carlo), and the calibration is differentiable: the calibrated parameters are a function of the quotes whose
derivatives come from the implicit function theorem, so the sensitivity of any contract priced with the calibrated model
to every quote is one gradient.</p>
</div>
<nav class="pkg-links">
<a class="pkg-btn" href="https://github.com/rcalxrc08/FinancialCalibrator.jl"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg> GitHub</a>
<a class="pkg-btn" href="https://github.com/rcalxrc08/FinancialCalibrator.jl/stargazers">★ Star</a>
<a class="pkg-btn" href="#Quick-start">Get started ↓</a>
<a class="pkg-btn" href="https://rcalxrc08.github.io/">All projects</a>
</nav>
```

![Implied volatilities of the quotes and of the calibrated Heston model](assets/figures/calibration-light.svg)![Implied volatilities of the quotes and of the calibrated Heston model](assets/figures/calibration-dark.svg)

## Features

- [`calibrate`](@ref) minimises the squared implied volatility errors of the out of the money quotes with any Optim
  optimizer, then polishes the solution with Levenberg–Marquardt to where the gradient of the loss vanishes, see
  [Calibration](calibration.md).
- [`calibrated_parameters`](@ref): the calibration as a differentiable function of the quoted volatilities, with
  derivatives from the implicit function theorem for ForwardDiff, Zygote (ChainRules) and Mooncake (a rule of its own),
  on the CPU and on the GPU, with the inner backends mixable, see [Sensitivity of the calibrated parameters to the
  quotes](@ref) and [Methods, backends and arrays](backends.md).
- Sensitivities of exotic contracts and positions to every quote of the surface, through the calibration and a Monte
  Carlo pricer, see [Sensitivities of exotics to the surface](exotics.md).
- Any model of FinancialMonteCarlo.jl and any pricing method of its `pricer` interface: FinancialFFT.jl's Fourier
  methods, closed forms, Monte Carlo, VibratoMonteCarlo.jl, see [Pricing methods](@ref).
- A choice of automatic differentiation backend for every derivative, inside and outside the calibration, see
  [Automatic differentiation](@ref).
- [`Parametrization`](@ref): unconstrained variables and fixed parameters.
- Float32 and GPU execution through CUDA.jl, derivatives included, see [Arrays](@ref).
- Market data: a DAX surface, and the free delayed quotes of the CBOE for US underlyings, see [Market data](#Market-data).

## Installation

FinancialFFT.jl is registered in a registry of its own, to be added once; the package is added from its repository:

```julia
] registry add https://github.com/rcalxrc08/LocalRegistry.git
] add https://github.com/rcalxrc08/FinancialCalibrator.jl
] add FinancialMonteCarlo, FinancialFFT
```

## Quick start

The package bundles an implied volatility surface of the DAX (22 August 2016). [`adapter`](@ref) parses it into the
`Underlying` of the models (spot and implied dividends) and the [`MarketData`](@ref) of the calibration:

The derivatives of a calibration (the Levenberg–Marquardt polish, the sensitivities) use ForwardDiff when it is loaded,
and finite differences otherwise; no automatic differentiation package is a dependency of FinancialCalibrator.

```@example quickstart
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT, ForwardDiff

und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
res = calibrate(model, LewisMethod(100.0, 4001), mkt)
(parameters = res.OptimalParam, rms_error = sqrt(2 * res.OptimalValue / count(!isnan, res.ModelVolatility)))
```

The sensitivity of the calibrated parameters to one quoted volatility:

```@example quickstart
i = CartesianIndex(4, 5)
bump(h) = (V = mkt.Volatility .+ zero(h); V[i] += h; V)
ForwardDiff.derivative(h -> calibrated_parameters(model, LewisMethod(100.0, 4001), mkt, bump(h)), 0.0)
```

## Market data

[`retriever`](@ref) reads the raw data of a calibration and [`adapter`](@ref) turns it into the `Underlying` and the
[`MarketData`](@ref). Besides the bundled DAX surface, the free delayed quotes of the CBOE give the implied volatility
surface of a US underlying: [`CboeInputStrategy`](@ref) reads a saved option chain, implies the forward and the dividends from put–call parity, and resamples the out of the money
volatilities on a moneyness grid.

[`CboeWebInputStrategy`](@ref) downloads the current chain instead, and can save it for later:

```julia
id = CalibrationID("SPX", Dates.format(today(), "dduyyyy"), "USD")
und, mkt = adapter(retriever(id, CboeWebInputStrategy(dir = "market_data")))   # downloaded, and saved in market_data/
und, mkt = adapter(retriever(id, CboeInputStrategy(dir = "market_data")))      # read again from the saved chain
```

## Relation to the other packages

[FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl) defines the models, the payoffs and the
`pricer` interface, [FinancialFFT.jl](https://github.com/rcalxrc08/FinancialFFT.jl) the Fourier pricing methods, and
[FinancialToolbox.jl](https://github.com/rcalxrc08/FinancialToolbox.jl) the implied volatility `blsimpv` and its
derivatives. FinancialCalibrator.jl adds the loss, the optimization and the derivatives of the calibration.
