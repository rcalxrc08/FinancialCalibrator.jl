# CLAUDE.md

FinancialCalibrator.jl calibrates option pricing models (Lévy processes, Heston, shifted lognormal mixture, ...) to
an implied volatility surface. Pricing is delegated to FinancialMonteCarlo (models, `pricer` interface) and
FinancialFFT (Fourier pricing methods); the optimization is done with Optim.

## Commands

```julia
julia --project=. -e 'using Pkg; Pkg.test()'     # all tests (about 3 minutes, the GPU part is skipped without a GPU)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Formatting uses the repo's `.JuliaFormatter.toml` (not a dependency, install it in a temporary environment):

```julia
using Pkg; Pkg.activate(temp = true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")
```

Run the formatter before committing. `FinancialFFT` lives in the public LocalRegistry
(`https://github.com/rcalxrc08/LocalRegistry.git`), CI adds it with the `localregistry` input of `julia-buildpkg`.

## Architecture

The public API is one function:

```julia
und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()), Float64)  # last argument: float type
calibrate(model, pricing_method, mkt, opt_method = NelderMead(), opt_options = Optim.Options(); verbose = false)
```

- `model` is a FinancialMonteCarlo process built on the `Underlying` that `adapter` returns. **Spot and dividends live
  in the model's underlying, never in `MarketData` and never as function arguments.**
- `pricing_method` decides how each smile is priced, through `pricer(model, ZeroRate(r), method, smile, mode)`: a
  FinancialFFT method (`CarrMadanMethod`, `CarrMadanLewisMethod`, `LewisMethod`, `CosMethod`) is numerical,
  `AnalyticMethod()` is closed form (only `ShiftedLogNormalMixture`). There are no calibrator structs: add a model/method
  pair by adding a `pricer` method.
- `MarketData{T}` holds maturities, zero rates, strikes and volatilities only. `Strikes` and `Volatility` can be GPU
  arrays (`Adapt.adapt(CuArray, mkt)`); `src/pricer.jl` has the `pricing_mode` hook the CUDA extension
  (`ext/FinancialCalibratorCUDAExt.jl`) overrides to select `CudaMode`.
- Floating point type is generic (`Float32` works): `MarketData`, `CalibratedData` are parametric and the signatures take
  `Real` / `AbstractVector{<:Real}`. Do not reintroduce `::Float64` annotations or Float64 literals in the pricing path.

Files in `src/` (included in this order by `FinancialCalibrator.jl`): `structs.jl` (input types, `MarketData`,
`implied_dividend`), `retriever_calibration.jl` (reads `data/json/`), `adapter_calibration.jl` (parsing, returns
`(underlying, mkt)`), `calibrated_data.jl` (`AnalyticMethod`, `CalibratedData`), `shifted_lognormal_mixture_pricer.jl`
(closed form and its `pricer` method), `pricer.jl`, `loss.jl`, `calibrate.jl`.

`loss` prices every maturity, inverts the prices to implied vols with `blsimpv` (using the model's spot and the
flat-equivalent dividend `implied_dividend(model.underlying, T)`), and sums the 2-norms against the market vols. It
returns a large penalty on invalid parameters, so exceptions thrown while optimizing are swallowed (see issue #3).

## Tests

`test/runtests.jl` (data, closed form, calibration, Float32), `test/models_methods.jl` (models × methods grid and
calibration of synthetic surfaces), `test/gpu.jl` (run only if `CUDA.functional()`). The tests define their own
`smile_prices` helper on top of `pricer`. `CarrMadanMethod` does not run on the GPU, and `CarrMadanLewisMethod` only
with a constant-dividend underlying (FinancialFFT limitations).

## Gotchas

- The sample data is `data/json/*Dax22Aug2016.json`; `examples/calibration_runner.jl` needs PyPlot, which is not a dependency.
- `FinancialMonteCarlo.set_parameters!` returns a **new** model, it does not mutate.
- `ImpliedCurve` prepends the point `0 => 0` so the integrated dividend over a maturity is not the spot value at that maturity.
- Editing files from Python on Windows: force UTF-8 (`PYTHONUTF8=1`), the sources contain non-ASCII characters.
- Commit messages end with the `Co-Authored-By` line configured for the session.
