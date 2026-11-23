# Code conventions and small gotchas

_Rules for every change._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

- `FinancialMonteCarlo.set_parameters!` returns a **new** model, it does not mutate.
- In a test, `q[[1, 2, 4, 5, 6]]` on a Jacobian is linear indexing: select rows with `q[[...], :]` (it once made the
  sensitivities look wrong).
- Return type assertions: none. The last one, in `analytic_prices` of a single option, went with FinancialMonteCarlo 0.8,
  whose mixtures hold tuples (the arity is in the type, #35). `loss`, `calibrate_cached` and the retrievers are inferred without
  one; JET (`@report_opt target_modules = (FinancialCalibrator,)`) is clean on `vol_errors`, `loss`, `loss_cache`, and on the polish with the backend `calibrate` passes
  (`polish_backend_for`, the ForwardDiff chunk fixed by the number of parameters of the model, #36).
