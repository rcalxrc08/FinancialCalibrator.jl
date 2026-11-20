# Stage 03: loss

The volatility errors of the quotes that enter the loss, for the parameters of a model.

## Inputs

| Load | Why |
|:--|:--|
| `loss.jl` | `loss_cache`, `vol_errors`, `maturity_errors`, `loss`, `numerical_failure` |
| `parametrization.jl` | `Parametrization`, `default_parametrization` per model |
| `backends.jl` | `default_backend()`: ForwardDiff when loaded, FiniteDiff otherwise |
| `ext/FinancialCalibratorMooncakeExt.jl` | the Mooncake rule of `loss`, when changing `loss` or `loss_value` |
| `references/loss.md` | the out of the money quotes with enough vega, why the loss is ½‖e‖², the GPU syncs |
| `references/autodiff.md` | which backends differentiate `vol_errors` and `loss`, Enzyme and Zygote traps |
| `references/dependencies.md` | before changing `backends.jl` |

## Process

- `loss_cache` builds once what does not depend on the parameters, and must not mutate arrays (Zygote traces it).
- `vol_errors` is generic in the number type of the parameters and of `vols` (a keyword); it throws on invalid
  parameters; `loss` returns `Inf` only for a `DomainError`, every other exception propagates.
- The `try` of `loss` stays out of what Zygote and Mooncake differentiate: `loss_value` is the loss without it, and the
  rules of `loss` (the `rrule` in `loss.jl`, the Mooncake one in `ext/`) run the `try` around its reverse pass (#40).
  Change the three together.
- One check of the prices per call (one GPU sync), `@inbounds` after the size check on the host.

## Outputs

`vol_errors(params, model, method, mkt, cache; vols)` and `loss(params, ...)`, inferred (`@inferred` in the tests, JET
clean). Checked by `test/runtests.jl` (`vol_errors`, the quotes left out, Float32, inference) and the AD tests.
