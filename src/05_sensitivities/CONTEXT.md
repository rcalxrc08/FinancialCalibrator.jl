# Stage 05: sensitivities

The calibrated parameters as a differentiable function of the quoted volatilities.

## Inputs

| Load | Why |
|:--|:--|
| `sensitivities.jl` | `CalibrationSolver`, `CalibrationConditions`, `implicit_calibration`, `calibrated_parameters` |
| `references/sensitivities.md` | backends verified, the methods whose derivatives are wrong (Cos on VG/NIG) |
| `references/autodiff.md` | the outer and inner backends, the Enzyme rules of the extension |
| `ext/CONTEXT.md` | when touching the Enzyme or Mooncake rules |

## Process

- The optimizer is never differentiated: ImplicitDifferentiation applies ∂x*/∂v = −(∂ₓ∇ₓL)⁻¹ ∂ᵥ∇ₓL at the solution.
- The derivatives are only as good as the Hessian: at an ill-conditioned optimum (the shifted mixture on DAX goes to the
  Bachelier limit, condition number 5e9) they depend on rounding, whatever the backend.

## Outputs

`calibrated_parameters(model, method, mkt, vols; conditions_backend, implicit_backends, ...)`. Checked by
`test/runtests.jl` (BS against 1/n, recalibration) and `test/models_methods.jl` (−H⁻¹∂ᵥ∇L, Zygote, Mooncake, Enzyme).
