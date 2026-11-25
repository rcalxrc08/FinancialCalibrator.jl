# Loss, quotes in the loss, parametrization

_Stage 03 reference._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

`Parametrization` is how a calibration searches over unconstrained variables and fixes parameters (`Parametrization()`
is the identity). The default is `default_parametrization(model)` (#24): exp/tanh/logistic per parameter, θ of VG and
NIG below the bound σ and κ set (`1/κ - σ²/2 - exp(x)`, `(1/κ - σ²)/2 - exp(x)`), mixture weights as a softmax with the
remainder as one more class, a `ShiftedProcess` the variables of its model then α = 1 - e^{-x} (`shift_from_variable`: FinancialMonteCarlo 0.8 throws a `DomainError` for α ≥ 1, and the default test checks variables ±3 around the start); identity for other models. Heston's λ is fixed at 0:
`get_parameters` returns every field, and the quotes cannot separate λ from κ (singular Hessian, no implicit
derivative). The optimizer and the polish work on `x`;
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
exact fit, unlike the old sum of per-maturity norms) and is `Inf` where the parameters cannot be priced
(`numerical_failure`: a `DomainError`, which is also what every FinancialMonteCarlo ≥ 0.8 constructor throws on invalid parameters, possible with `Parametrization()`; FinancialMonteCarlo#71, #31: an `AssertionError` or a `DimensionMismatch` is a bug and propagates);
any other exception propagates (#3, #24; the polish rejects the same exceptions and rethrows the rest), and so does a numerical failure where the plain values of the parameters can be priced (`priceable`: a failure of the derivatives only, #42; one more evaluation of the errors, on the failure path only). The loop over the maturities, `maturity_errors`, sits behind
`FinancialMonteCarlo.set_parameters!`, which is not inferred; `loss` converts to `promote_type(eltype(params), T)` and is
`@inferred` in the tests, as the closed-form `pricer` (which writes into an array of the price type computed by `analytic_price_type`, `analytic_prices`; the kernel of a mixture has its arity as a type parameter, known from the tuples of the model).

- `maturity_errors` prices all the maturities first (`model_prices`, one `reduce(vcat, ...)` of a comprehension), checks
  the prices once (`check_prices`: one reduction read back to the host, i.e. one GPU synchronization instead of one per
  maturity), then one `blsimpv` broadcast over all the quotes and one subtraction, with the strikes, rates, maturities,
  dividends, call flags and positions in the volatility matrix of all the maturities in `cache.flat` (built once by
  `loss_cache`; `cache.active` are the maturities with quotes). Per-maturity cache fields stay for `calibrate` and
  `model_volatilities` (one maturity, with its own check, used by the tests). `loss_cache` must not mutate any array
  (`copyto!`, `fill!`): Zygote traces it through `implicit_calibration` when it differentiates `calibrated_parameters`
  and fails ("Mutating arrays is not supported"). The positions stay on the host; `vols[flat.positions]` is `@inbounds`
  after `vol_errors` checks `size(vols)` on the host: the bounds check of GPUArrays is an `all` reduction read back to
  the host, one more sync. Measured on Heston/Lewis DAX, MX150, interleaved rounds (#23 comment): calibration 2.44 s
  (check per maturity) -> 1.81 s (one check) -> 1.64 s (+ `@inbounds`) -> 1.30 s once FinancialMonteCarlo stops
  re-validating each smile in `to_mode` (`minimum(K)` on the device, 9 syncs per loss, FinancialMonteCarlo#73); the
  host takes ~1.3 s. Zero host checks (invalid prices masked to NaN on the device) gave nothing more. After that the
  GPU loss is launch-bound: 140 launches per `vol_errors` (FinancialFFT#92).
- (history) `maturity_errors` was a loop with a concrete accumulator because a `map` closure was not inferred with dual numbers and made
  the loss `Any` (tested with `@inferred` on duals in the parameters and in the quotes).
