# Optimizer and Levenberg-Marquardt polish

_Stage 04 reference._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

`calibrate` builds the cache and calls `calibrate_cached`, which calls `optimal_variables_and_errors` (`OptimalValue` and
`ModelVolatility` come from the polish's last residuals, no re-pricing; #22): Optim's `optimize` with the default
options `DEFAULT_OPTIONS = Optim.Options(g_tol = 1e-6)`, which hand over to the polish (Optim's `1e-8` crawled: Kou
2.7 s -> 1.4 s on DAX, same optimum for 7 models; `FULL_OPTIONS` with `polish = false`), and
`autodiff` (an ADTypes backend, used by gradient optimizers), then with `polish = true` `levenberg_marquardt` on
`vol_errors` (Jacobian by `polish_backend` through DifferentiationInterface; steps that throw or do not decrease the sum of
squares rejected, `step_accepted`: a step is taken if it lowers the sum of squares (`Σ (e_new - e)(e_new + e)`, no
cancellation) or, within sqrt(eps) of it, halves the gradient Jᵀe; near the optimum the noise of the implied vols
(~1e-14, ±1e-15 on the sum) hides the decrease a step promises (6e-18 on DAX/BS), and the GPU polish stopped 9e-10
short of σ* = mean of the quotes; a step accepted on the gradient starts a phase in which every step must halve the
last gradient and stay within sqrt(eps) of the sum where it began (with a FiniteDiff Jacobian the two criteria
alternated for all 100 iterations); a rejected step below `xtol`, a non-finite Jacobian (Kou on DAX), or (#37) three rejections in a row once the
decrease the linear model promises, `-(gᵀδ + δᵀAδ/2)`, is below `sqrt(eps)` times the loss, ends the polish (at the floor
the steps are ~1e-9, above `xtol`, and μ needed ~30 rejections to pass 1e12; measured 2026-11-26 on DAX: Heston
`LewisMethod(100, 4001)` 37 -> 23 calls of the errors, 0.43 -> 0.27 s, NIG 33 -> 27, BS 9 -> 9, end points bit-identical);
the Jacobian is recomputed only after an accepted step; with J on the GPU, JᵀJ and Jᵀe are formed there and the p × p solve is on the host, timings in the
comment above the function). The polish is what makes the
end point stationary (|∇L| ~ 1e-8 on DAX instead of NelderMead's ~1e-3).

- Time split of a default calibration (Heston/Lewis 4001, DAX, host): Nelder–Mead 60% (142 losses, hands over at
  L = 4.1e-3 vs 2.35e-3), polish 40% (17 linearly converging steps, then ~28 trials at the rounding floor, #37),
  everything else < 1%. Polish only: 0.95 s vs 1.88 s, same optimum (#33). `LewisMethod(100, 2001)` halves the cost
  at 7e-7 in the vols (#10). Scripts were in the session scratchpad, not in the repository.
- Default arguments of `calibrate`/`calibrated_parameters` are the constants `DEFAULT_OPTIMIZER`/`DEFAULT_OPTIONS`, not
  `NelderMead()`: a default argument is evaluated in the method Mooncake traces, and `NelderMead()` builds `Dict(kwargs)`,
  whose constructor has a try/catch on Julia 1.10 (Mooncake fails). Do not construct anything with keywords there.

## When the optimizer stops at its limit (#46)

`optimal_variables_and_errors` also returns the Optim result. `calibrate_cached` keeps `Optim.converged`,
`Optim.iterations` and `Optim.f_calls` in `CalibratedData` (`Converged`, `Iterations`, `LossCalls`) and warns when
`Optim.termination_code` is `Iterations` or `Time`, with a hint about an inaccurate pricing method when the optimizer
took more than 20 losses per iteration (a line search with wrong derivatives: LBFGS on Kou with `CosMethod(256)` on DAX
spent 1000 iterations and 49486 losses, 73 s, FinancialFFT#101). The warning is in `calibrate` only, not in the solver of
`calibrated_parameters`, which stays inferred (checked 2026-11-26: `optimal_variables_and_errors` inferred with the
result type).

## Where the optimizer stops at an unpriceable point (#38)

`optimal_variables_and_errors` prices the optimizer's end point once (`numerical_or_nothing`): if it cannot be priced
(a numerical failure such as a negative model price), the polish is skipped and `calibrate` returns the `Inf` result,
with or without the polish; otherwise the polish starts from those residuals, so the surface is not priced twice. Before,
the polish priced the point itself, outside its failure guard, and the `DomainError` of the loss came out.

## The polish is inferred: the ForwardDiff chunk from the model (#36)

`DI.prepare_jacobian(errors, AutoForwardDiff(), x)` let ForwardDiff pick its chunk from `length(x)` at run time: the
prepared Jacobian was a `UnionAll` over the chunk, and everything the Levenberg–Marquardt loop computes from it (`J`,
`A`, `g`, `D`, the step, the phase of `step_accepted`) dispatched at run time, 21 reports of JET in FinancialCalibrator
(`errors` itself and `DI.jacobian` with a concrete preparation were inferred). The number of parameters of a model is a
constant of its type, `FinancialMonteCarlo.parameter_count` (FinancialMonteCarlo 0.9.1), so `calibrate` passes
`polish_backend_for(polish_backend, model, x)`: `AutoForwardDiff{min(N, 12)}` when the backend has no chunk size and the
parametrization keeps the number of variables, the backend as given otherwise (a chunk larger than the input is an
error). `normal_equations` takes `errors::F where F`, since it only passes it on (Julia would compile it for
`::Function`). No function barrier.

Measured 2026-11-23 (i7-8550U, Julia 1.13.4; DAX, shifted lognormal mixture, `AnalyticMethod`, polish from the start
point; JET `report_opt` on the polish; before and after in interleaved processes, two rounds, BenchmarkTools):

| | JET reports (in FinancialCalibrator) | min | allocations |
|:--|--:|--:|--:|
| before (run-time chunk) | 123 (21) | 14.5-15.4 ms | 11600 |
| after (`AutoForwardDiff{6}`) | 40 (0) | 14.3-15.0 ms | 10462 |

Same end point and loss. The 40 reports left are in Base (`show` and string handling of the error and `@debug` paths).
