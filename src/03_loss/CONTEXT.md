# Stage 03: loss

The volatility errors of the quotes that enter the loss, for the parameters of a model.

## Inputs

| Load | Why |
|:--|:--|
| `loss.jl` | `loss_cache`, `vol_errors`, `maturity_errors`, `loss`, `numerical_failure`, `priceable` |
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
  Change the three together. A numerical failure is `Inf` only if `priceable` (the plain values) fails too (#42); in
  `loss` that check stays after the `try`, never in the `catch` (Enzyme runtime activity).
- One check of the prices per call (one GPU sync), `@inbounds` after the size check on the host.

## Outputs

`vol_errors(params, model, method, mkt, cache; vols)` and `loss(params, ...)`, inferred (`@inferred` in the tests, JET
clean). Checked by `test/runtests.jl` (`vol_errors`, the quotes left out, Float32, inference) and the AD tests.

## Verify (Kaimon)

Run in a session of this project after a change here (`ex(q = false, ...)`; setup: `using FinancialCalibrator, FinancialMonteCarlo, FinancialToolbox, Test, Dates; const FC = FinancialCalibrator`). Observed 2026-11-21, Julia 1.13.4, FinancialMonteCarlo 0.9.0.

| Check | Expect |
|:--|:--|
| `cache = FC.loss_cache(BlackScholesProcess(0.2, und), AnalyticMethod(), mkt)` on DAX; `sum(length, cache.quoted)` | 99 of 108 quotes (9 left out for low vega) |
| `@inferred FC.loss([0.2], model, AnalyticMethod(), mkt)`, `@inferred vol_errors([0.2], model, AnalyticMethod(), mkt, cache)` | `Float64`, `Vector{Float64}` |
| `JET.@report_opt target_modules = (FC,)` on `vol_errors`, `FC.loss(..., cache)`, `FC.loss_cache` | 0 reports each (without `target_modules`, `loss` reports ~40 runtime dispatches in Base's exception and `show` paths of its `catch`: not ours) |
