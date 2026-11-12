# CLAUDE.md

FinancialCalibrator.jl calibrates option pricing models (Lévy processes, Heston, shifted lognormal mixture, ...) to
an implied volatility surface. Pricing is delegated to FinancialMonteCarlo (models, `pricer` interface) and whatever
method package the user loads (FinancialFFT, VibratoMonteCarlo; FinancialFFT is a test and docs dependency only); the
optimization is done with Optim.

Dependencies, by design: **no package that differentiates is a hard dependency**. ADTypes and DifferentiationInterface
are interfaces (and hard dependencies of Optim through NLSolversBase anyway); ImplicitDifferentiation (the rules of
`calibrated_parameters`) is a hard dependency by the author's choice. ForwardDiff, Zygote, Enzyme, Mooncake are not:
`default_backend()` (`src/backends.jl`) is `AutoForwardDiff()` when DifferentiationInterface's ForwardDiff extension is
loaded (i.e. the user loaded ForwardDiff) and `AutoFiniteDiff()` otherwise (FiniteDiff comes with NLSolversBase); every
backend keyword defaults to it (`DI.check_available`, not `Base.get_extension`: the latter takes a lock that Zygote
cannot differentiate through the keyword defaults). `ext/FinancialCalibratorMooncakeExt.jl` (Mooncake + ChainRulesCore) gives Mooncake a
rule for `implicit_variables`: ImplicitDifferentiation's ChainRules rrule needs a `RuleConfig`, which Mooncake's
`@from_rrule` does not pass, so the extension calls it with a config without modes (the backends are always explicit in
`calibrated_parameters`, the config is never used) and moves the cotangent of the quotes itself. The setup of the implicit function is `implicit_calibration`, marked
`@zero_derivative` there: otherwise Mooncake traces it and fails on the device-to-host copies of the cache on the GPU.
AD support (CPU and GPU timings, failures) is the table in `docs/src/backends.md`: ForwardDiff, Zygote, Mooncake work
(Mooncake also on the GPU); Mooncake as inner `x` fails on the GPU (dual numbers broadcast over a CuArray); Enzyme
fails in both modes (reverse: internal assertion; forward: ImplicitDifferentiation's Enzyme rule has no method for the
config DI passes). Removed as unused: DelimitedFiles (with the dead `web_retriever` and
`WebFileInput`, #15), FFTW, Random, FinancialFFT.

## Commands

```julia
julia --project=. -e 'using Pkg; Pkg.test()'     # all tests (about 10 minutes, 15 with the GPU part, skipped without a GPU)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=docs -e 'using Pkg; Pkg.instantiate()'; julia --project=docs docs/make.jl   # the documentation (a few minutes)
```

The documentation is Documenter.jl, in the style of FinancialFFT.jl (same `docs/src/assets/style.css` and `figures.css`,
hero on `index.md`, figures computed at build time by `docs/figures.jl` in a light and a dark variant, from the
`@example` blocks of the pages run in a module of their own). `.github/workflows/DeployPage.yml` builds it and publishes it
with `actions/deploy-pages` (the Pages source of the repository must be "GitHub Actions"); the Franklin `page/` site was
removed. The logo and favicon are drawn by `etc/logo.jl` (Luxor, run in a temporary environment): a bird whose wings are a
Heston vol surface, a smile fitted to quotes, loss level sets with the Levenberg–Marquardt path; `docs/make.jl` copies them
into the assets.

Formatting uses the repo's `.JuliaFormatter.toml` (not a dependency, install it in a temporary environment):

```julia
using Pkg; Pkg.activate(temp = true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")
```

Run the formatter before committing. `FinancialFFT` lives in the public LocalRegistry
(`https://github.com/rcalxrc08/LocalRegistry.git`), CI adds it with the `localregistry` input of `julia-buildpkg`.

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
  `AnalyticMethod(; mode)` is closed form (only `ShiftedLogNormalMixture`; it has a `mode` field like every method). There are no calibrator structs: add a model/method
  pair by adding a `pricer` method.
- `MarketData{T}` holds maturities, zero rates, strikes and volatilities only. `Strikes` and `Volatility` can be GPU
  arrays (`Adapt.adapt(CuArray, mkt)`); `src/pricer.jl` has the `calibration_method(method, strikes)` hook (the method as
  given), which the CUDA extension (`ext/FinancialCalibratorCUDAExt.jl`) overrides for a `CuArray` with
  `with_mode(method, CudaMode())`; `loss` and `calibrate` apply it. A method already in `CudaMode` moves host strikes to the
  device itself (`FinancialMonteCarlo.to_mode` in the pricers).
- Floating point type is generic (`Float32` works): `MarketData`, `CalibratedData` are parametric and the signatures take
  `Real` / `AbstractVector{<:Real}`. Do not reintroduce `::Float64` annotations or Float64 literals in the pricing path.

Files in `src/` (included in this order by `FinancialCalibrator.jl`): `structs.jl` (input types, `MarketData`,
`implied_dividend`), `retriever_calibration.jl` (reads `data/json/`), `cboe.jl` (`CboeInputStrategy` from a file, `CboeWebInputStrategy` downloading the current chain, `cboe_snapshot`:
CBOE delayed quotes, #18), `adapter_calibration.jl` (parsing, returns
`(underlying, mkt)`), `calibrated_data.jl` (`AnalyticMethod`, `CalibratedData`), `shifted_lognormal_mixture_pricer.jl`
(closed form and its `pricer` method), `pricer.jl`, `loss.jl`, `parametrization.jl`, `calibrate.jl`, `sensitivities.jl`.

`Parametrization` (default: identity both ways) is how a calibration searches over unconstrained variables and fixes
parameters: `get_parameters` returns every field, so Heston includes λ, which the quotes cannot separate from κ (singular
Hessian, no implicit derivative) unless a parametrization fixes it. The optimizer and the polish work on `x`;
`CalibratedData.OptimalParam` and `calibrated_parameters` return `to_params(x*)`.

`loss_cache(model, method, mktData; min_vega = 1e-3)` builds once per calibration what does not depend on the parameters:
for each maturity the indices of the quotes that enter the loss (`quoted`, on the host), their strikes and call flags, the
smile, the rate and the dividend, and the method moved to the mode of the strikes (`calibration_method`), so a
`MonteCarloConfiguration` (whose `with_mode` rebuilds its random state) or a Vibrato method can be the `pricing_method`
too. The quotes are **out of the money** (puts below the forward `S0 exp((r - d) T)`, calls above it: an in the money
implied vol is ill-conditioned) with Black–Scholes vega at the quoted vol `>= min_vega * S0` (on DAX 9 of 108 are left
out; the one at T = 0.17, moneyness 1.75 is priced ~1e-10 by Cos and made the loss jitter at 1e-6, so no optimizer reached
a stationary point). `ModelVolatility` of `CalibratedData` is `NaN` at the quotes left out.

`vol_errors` returns model minus quoted vols of those quotes, maturity by maturity, generic in the number type of the
parameters and of `vols` (a keyword, so the quotes can be dual numbers while `MarketData` stays `AbstractFloat`). It throws
on invalid parameters and a `DomainError` on a non-positive price. `loss` is half its sum of squares (smooth also at an
exact fit, unlike the old sum of per-maturity norms) and returns a large penalty on any exception, so exceptions thrown
while optimizing are swallowed (see issue #3). The loop over the maturities, `maturity_errors`, sits behind
`FinancialMonteCarlo.set_parameters!`, which is not inferred; `loss` converts to `promote_type(eltype(params), T)` and is
`@inferred` in the tests, as the closed-form `pricer` (which writes into an array of the price type, `slnm_prices`).

`calibrate` builds the cache and calls `calibrate_cached`, which calls `optimal_variables`: Optim's `optimize` with
`autodiff` (an ADTypes backend, used by gradient optimizers), then with `polish = true` `levenberg_marquardt` on
`vol_errors` (Jacobian by `polish_backend` through DifferentiationInterface; steps that throw or do not decrease the sum of
squares rejected; with J on the GPU, JᵀJ and Jᵀe are formed there and the p × p solve is on the host, timings in the
comment above the function). The polish is what makes the
end point stationary (|∇L| ~ 1e-8 on DAX instead of NelderMead's ~1e-3).

`sensitivities.jl`: `calibrated_parameters` (the solver and the conditions are structs there; `implicit_variables` is
the call of the implicit function, a separate function for the Mooncake rule) wraps the calibration in `ImplicitDifferentiation.ImplicitFunction`:
`CalibrationSolver` (calls `optimal_variables` on plain numbers, with the cache built on `mktData`, so the solver and the
conditions use the same quotes) and `CalibrationConditions` (∇ₚL by `DI.gradient` with `conditions_backend`), which
ImplicitDifferentiation differentiates with `implicit_backends` (x: the quotes, y: the parameters), `MatrixRepresentation`
and `DirectLinearSolver`, with `strict = Val(false)`: Zygote's cotangent of `sum(p)` is a lazy vector, not the
`Vector` the pullback was prepared with, and the strict check of DifferentiationInterface rejected it. All backends default to ForwardDiff (few parameters). The outer backend is free: ForwardDiff
duals, and the ChainRules rrule (Zygote tested, equal to ForwardDiff to 5e-16; Mooncake via `@from_chainrules`). With
Zygote outside, `x = AutoMooncake` is also exact (3.9e-16) but 160 s against 22 s on 15 quotes, mostly compilation;
`x = AutoZygote` fails (Zygote cannot differentiate the inner `ForwardDiff.gradient`: tag ordering error). Checked on DAX against central differences of recalibrations: BS with Cos (∂σ*/∂vᵢ = 1/n exactly, n the quotes in the loss), VG and NIG
with Lewis to ~1e-3. **Not with `CosMethod` on VG/NIG**: FinancialFFT holds the Cos truncation range constant under AD,
and where the series converges slowly (small T/κ) its derivatives are not those of its values (FinancialFFT#64), so the
optimizer stops at |∇L| ~ 1e-3 and the implicit derivatives are wrong. Use Lewis or Carr-Madan-Lewis there.
On the GPU (quotes as `CuArray`) `calibrate`, the ForwardDiff derivative and the Zygote gradient (a `CuMatrix`) of
`calibrated_parameters` work and match the host (BS 1e-13, VG/Lewis 1e-7); timings in `docs/src/backends.md`.

`cboe.jl`: `CboeWebInputStrategy` and `cboe_snapshot` download (`cboe_download`) `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` with the
`Downloads` stdlib (no HTTP.jl) and saves it (in `pwd()` by default: the package directory is read-only once installed) as `CboeOptions_<underlying><ddMonyyyy>.json`; `retriever(id,
CboeInputStrategy(; rate, moneyness, root, dir))` turns a saved chain into the JSON strings of the existing schema, so
`adapter` is reused: OCC symbols parsed by `parse_occ`, quotes with `bid > 0` and `1e-3 < iv < 5`, the forward of each
expiry by put–call parity (median over the 6 strikes nearest to the spot), the dividend `r - log(F/S0)/T`, a flat
`rate`, out of the money vols interpolated linearly in moneyness on the grid (expiries not covering it are dropped,
since the adapter needs one grid for all maturities, #11). At least 4 expiries (Dierckx cubic splines). Yahoo's option
chain needs a cookie and crumb that it refuses to non-browser clients (401/429), so it is not implemented. No real chain is in the repository (too heavy): `test/cboe.jl` builds a synthetic chain
in the CBOE format (Black–Scholes with a skew, bid/ask 1% around the price, so parity is exact) in a temporary
directory; `test/local/cboe_spx.jl` (NIG calibrated to the real SPX chain and its implicit derivative against
recalibration) runs only when `data/json/real/CboeOptions_SPX12Nov2026.json` exists, a git-ignored folder with the
downloaded snapshots and a README. The adapter now selects the volatility rows with the same mask as the maturities (#2).

## Tests

`test/runtests.jl` (data, closed form, calibration, `vol_errors` and the quotes left out, stationarity on DAX,
`calibrated_parameters` on DAX against 1/n and recalibration, Float32), `test/models_methods.jl` (models × methods grid,
calibration of synthetic surfaces, and the full Jacobian of `calibrated_parameters` against -H⁻¹∂ᵥ∇ₚL and recalibration
for BS/Cos and VG/Lewis, and its Zygote gradient against Jᵀw and with a lazy cotangent; Zygote is a test-only
dependency), `test/financialfft_example.jl` (the calibration page of FinancialFFT run verbatim, Heston by hand-written
Levenberg–Marquardt and H⁻¹Jᵀ, against `calibrate` and the ForwardDiff Jacobian of `calibrated_parameters` with the same
`Parametrization`: parameters 1e-8, sensitivities 1e-6; and the gradient of a Monte Carlo Asian option through the
calibration against the chain rule and recalibration with bumps of 1e-6, larger ones cross the kinks of the variance
floor since the calibrated Heston violates Feller), `test/cboe.jl` (a synthetic CBOE chain: parsing, filters, parity, the surface, and the adapter with a maturity
below the threshold, #2), `test/local/cboe_spx.jl` (local only, see above), `test/aqua.jl` (Aqua.test_all), `test/gpu.jl` (run only if
`CUDA.functional()`, includes `calibrated_parameters` on device quotes). The tests define their own
`smile_prices` helper on top of `pricer`. Every FinancialFFT method runs on the GPU (FinancialFFT ≥ 0.5). The compat is FinancialFFT 0.6 and FinancialMonteCarlo 0.7: their breaking changes needed no change here (`ImpliedCurve` still takes vectors, smiles have `AbstractVector` strikes).

## Gotchas

- The sample data is `data/json/*Dax22Aug2016.json`; `examples/calibration_runner.jl` needs PyPlot, which is not a dependency.
- `FinancialMonteCarlo.set_parameters!` returns a **new** model, it does not mutate.
- `ImpliedCurve` prepends the point `0 => 0` so the integrated dividend over a maturity is not the spot value at that maturity.
- Editing files from Python on Windows: force UTF-8 (`PYTHONUTF8=1`), the sources contain non-ASCII characters.
- In a test, `q[[1, 2, 4, 5, 6]]` on a Jacobian is linear indexing: select rows with `q[[...], :]` (it once made the
  sensitivities look wrong).
- Monte Carlo prices through FinancialMonteCarlo reseed at every call, so their AD gradient is the pathwise derivative
  on fixed paths: right for smooth payoffs, biased for barriers and binaries (the knock-out is not differentiated).
- Commit messages end with the `Co-Authored-By` line configured for the session.
