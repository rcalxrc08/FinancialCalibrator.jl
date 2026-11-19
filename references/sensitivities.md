# calibrated_parameters: implicit derivatives

_Stage 05 reference._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

`src/05_sensitivities/sensitivities.jl`: `calibrated_parameters` (the solver and the conditions are structs there; `implicit_variables` is
the call of the implicit function, a separate function for the Mooncake rule) wraps the calibration in `ImplicitDifferentiation.ImplicitFunction`:
`CalibrationSolver` (calls `optimal_variables` on plain numbers, with the cache built on `mktData`, so the solver and the
conditions use the same quotes) and `CalibrationConditions` (∇ₚL by `DI.gradient` with `conditions_backend`), which
ImplicitDifferentiation differentiates with `implicit_backends` (x: the quotes, y: the parameters), `MatrixRepresentation`
and `DirectLinearSolver`, with `strict = Val(false)`: Zygote's cotangent of `sum(p)` is a lazy vector, not the
`Vector` the pullback was prepared with, and the strict check of DifferentiationInterface rejected it. All backends default to ForwardDiff (few parameters). The outer backend is free: ForwardDiff
duals, and the ChainRules rrule (Zygote tested, equal to ForwardDiff to 5e-16; Mooncake via `@from_chainrules`). With
Zygote outside, `x = AutoMooncake` is also exact (3.9e-16) but 160 s against 22 s on 15 quotes, mostly compilation;
`x = AutoZygote` fails (Zygote cannot differentiate the inner `ForwardDiff.gradient`: tag ordering error). Checked on DAX against central differences of recalibrations: BS with Cos (∂σ*/∂vᵢ = 1/n exactly, n the quotes in the loss), VG and NIG
with Lewis to ~1e-3. **Not with `CosMethod` on VG/NIG**: FinancialFFT holds the Cos truncation range constant under AD,
and where the series converges slowly (small T/κ) its derivatives are not those of its values (FinancialFFT#64), so the
optimizer stops at |∇L| ~ 1e-3 and the implicit derivatives are wrong. Use Lewis or Carr-Madan-Lewis there.
On the GPU (quotes as `CuArray`) `calibrate`, the ForwardDiff derivative and the Zygote gradient (a `CuMatrix`) of
`calibrated_parameters` work and match the host (BS 1e-13, VG/Lewis 1e-7); timings in `docs/src/backends.md`.
