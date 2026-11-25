# Stage 04: calibration

The optimizer, then the Levenberg–Marquardt polish that makes the end point stationary.

## Inputs

| Load | Why |
|:--|:--|
| `calibrate.jl` | `calibrate`, `calibrate_cached`, `optimal_variables_and_errors`, `levenberg_marquardt`, `step_accepted` |
| `calibrated_data.jl` | `CalibratedData` |
| `references/optimization.md` | the defaults, the acceptance rules of the polish, the time split |
| `references/loss.md` | what the polish minimizes |

## Process

- Defaults are constants (`DEFAULT_OPTIMIZER`, `DEFAULT_OPTIONS`): nothing built with keywords in default arguments
  (Mooncake traces them).
- The polish prepares its Jacobian once (`polish_backend` through DifferentiationInterface), so the solver is inferred:
  the Enzyme rules of stage 05 need it. A ForwardDiff backend without a chunk size gets the chunk from
  `FinancialMonteCarlo.parameter_count` (`polish_backend_for`, #36), so the loop is inferred too.

## Outputs

`calibrate(model, und, method, mkt, opt_method, opt_options; parametrization, autodiff, polish, polish_backend, min_vega)`
returns `CalibratedData` (`OptimalParam`, `OptimalValue`, `ModelVolatility`, `NaN` at the quotes left out). Checked by
`test/runtests.jl` (stationarity on DAX) and `test/models_methods.jl` (synthetic surfaces recovered).

## Verify (Kaimon)

Run in a session of this project after a change here (`ex(q = false, ...)`; setup: `using FinancialCalibrator, FinancialMonteCarlo, FinancialToolbox, Test, Dates; const FC = FinancialCalibrator`). Observed 2026-11-21, Julia 1.13.4, FinancialMonteCarlo 0.9.0.

| Check | Expect |
|:--|:--|
| `calibrate(bs, AnalyticMethod(), MarketData([0.5, 1.0, 2.0], Date(2020) .+ Day.([182, 365, 730]), fill(0.01, 3), 70.0:10.0:130.0, fill(0.25, 3, 7)))` | `OptimalParam ≈ [0.25]` (rel. 1e-15), `OptimalValue` ~1e-31 |
| `calibrate(BlackScholesProcess(0.2), und, AnalyticMethod(), mkt)` on DAX | `OptimalParam ≈ [0.2010]`, `OptimalValue ≈ 0.0492`, 9 `NaN` in `ModelVolatility` (the quotes left out) |
