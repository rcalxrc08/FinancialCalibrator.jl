# Stage 02: pricing

A model and a method price the smiles of one maturity.

## Inputs

| Load | Why |
|:--|:--|
| `pricer.jl` | `calibration_method(method, strikes)`: the method as given; the CUDA extension switches it to `CudaMode` for `CuArray` strikes |
| `analytic/analytic_method.jl` | `AnalyticMethod`, the generic closed-form `pricer`, `LognormalMixtureKernel`, `stateless_prices` (Mooncake on the device) |
| `analytic/black_scholes.jl`, `analytic/shifted_lognormal_mixture.jl` | `analytic_price_type` and `analytic_kernel` per model |
| `references/pricing.md` | what each file defines |
| `references/autodiff.md` | the kernel must stay inferred under Zygote (no branches, no `oftype`, `@generated` sums) |

## Process

- There are no calibrator structs: a model/method pair is a `pricer` method. A new closed form is two methods,
  `analytic_price_type` and `analytic_kernel`, for its model.
- Closed forms come from FinancialToolbox (`blsprice`, `blkprice`), never written by hand.
- Generic in the number type; the kernel is an isbits callable, so a GPU broadcast takes it. A new field of
  `LognormalMixtureKernel` also goes into `MixturePrice`/`stateless_prices`, which Mooncake differentiates on the device.

## Outputs

`pricer(model, ZeroRate(r), method, smile)` returns one price per strike, on the device of the strikes. Checked by
`test/runtests.jl` (closed form against Black–Scholes), `test/models_methods.jl` (models × methods), `test/gpu.jl`.

## Verify (Kaimon)

Run in a session of this project after a change here (`ex(q = false, ...)`; setup: `using FinancialCalibrator, FinancialMonteCarlo, FinancialToolbox, Test, Dates; const FC = FinancialCalibrator`). Observed 2026-11-21, Julia 1.13.4, FinancialMonteCarlo 0.9.0.

| Check | Expect |
|:--|:--|
| `bs = BlackScholesProcess(0.2, FinancialMonteCarlo.Underlying(100.0, 0.03))`; `smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, strikes, strikes .>= 100)` with `strikes = 40.0:10.0:250.0`; `@inferred pricer(bs, ZeroRate(0.01), AnalyticMethod(), smile)` | `Vector{Float64}`, equal to `blsprice.(...)` (max relative difference 0.0) |
| with CUDA, Mooncake and the bench environment: `DI.gradient(q -> FC.loss(q, slnm, AnalyticMethod(), adapt(CuArray, mkt)), AutoMooncake(; config = nothing), ps)` on DAX, `slnm = ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3), und), -0.1)` | ForwardDiff on the host to 1e-15 (1e-14 for Black–Scholes); `Base.get_extension(FC, :FinancialCalibratorCUDAMooncakeExt)` loaded |
