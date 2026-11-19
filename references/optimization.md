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
alternated for all 100 iterations); a rejected step below `xtol` or a non-finite Jacobian (Kou on DAX) ends the polish;
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
