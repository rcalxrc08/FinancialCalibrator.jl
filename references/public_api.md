# Public API and the data flow

_The functions a user calls, and the rules on where data lives._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

## Architecture

The public API:

```julia
und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()), Float64)  # last argument: float type
calibrate(model, pricing_method, mkt, opt_method = NelderMead(), opt_options = Optim.Options(); parametrization = Parametrization(), autodiff = AutoForwardDiff(), polish = true, polish_backend = AutoForwardDiff(), min_vega = 1e-3, verbose = false)
calibrated_parameters(model, pricing_method, mkt, vols = mkt.Volatility, opt_method, opt_options; parametrization, autodiff, polish_backend, conditions_backend, implicit_backends = (x = ..., y = ...), min_vega)  # differentiable in vols
Parametrization(to_params, from_params)   # the variables of the search: x -> model parameters, and its inverse
vol_errors(params, model, pricing_method, mkt, cache = loss_cache(...); vols = mkt.Volatility)                       # residuals of the loss
```

- `model` is a FinancialMonteCarlo process built on the `Underlying` that `adapter` returns. **Spot and dividends live
  in the model's underlying, never in `MarketData` and never as function arguments.**
- `pricing_method` decides how each smile is priced, through `pricer(model, ZeroRate(r), method, smile)` (four arguments: since FinancialMonteCarlo 0.6 the execution mode is part of the method, `FinancialMonteCarlo.with_mode(method, mode)`): a
  FinancialFFT method (`CarrMadanMethod`, `CarrMadanLewisMethod`, `LewisMethod`, `CosMethod`) is numerical,
  `AnalyticMethod(; mode)` is closed form (`BlackScholesProcess`, `LogNormalMixture` with tuple parameters, and `ShiftedProcess` of either: the displaced diffusion and the shifted lognormal mixture; it has a `mode` field like every method). There are no calibrator structs: add a model/method
  pair by adding a `pricer` method.
- `MarketData{T}` holds maturities, zero rates, strikes and volatilities only. `Strikes` and `Volatility` can be GPU
  arrays (`Adapt.adapt(CuArray, mkt)`); `src/02_pricing/pricer.jl` has the `calibration_method(method, strikes)` hook (the method as
  given), which the CUDA extension (`ext/FinancialCalibratorCUDAExt.jl`) overrides for a `CuArray` with
  `with_mode(method, CudaMode())`; `loss` and `calibrate` apply it. A method already in `CudaMode` moves host strikes to the
  device itself (`FinancialMonteCarlo.to_mode` in the pricers).
- Floating point type is generic (`Float32` works): `MarketData`, `CalibratedData` are parametric and the signatures take
  `Real` / `AbstractVector{<:Real}`. Do not reintroduce `::Float64` annotations or Float64 literals in the pricing path.
