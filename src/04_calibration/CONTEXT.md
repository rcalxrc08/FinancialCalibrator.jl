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
  the Enzyme rules of stage 05 need it.

## Outputs

`calibrate(model, method, mkt, opt_method, opt_options; parametrization, autodiff, polish, polish_backend, min_vega)`
returns `CalibratedData` (`OptimalParam`, `OptimalValue`, `ModelVolatility`, `NaN` at the quotes left out). Checked by
`test/runtests.jl` (stationarity on DAX) and `test/models_methods.jl` (synthetic surfaces recovered).
