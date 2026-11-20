# Tests

_What each test file covers and how long it takes._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

`test/runtests.jl` (data, closed form, calibration, `vol_errors` and the quotes left out, stationarity on DAX,
`calibrated_parameters` on DAX against 1/n and recalibration, Float32), `test/models_methods.jl` (models × methods grid,
calibration of synthetic surfaces, and the full Jacobian of `calibrated_parameters` against -H⁻¹∂ᵥ∇ₚL and recalibration
for BS/Cos and VG/Lewis, and its Zygote gradient against Jᵀw and with a lazy cotangent; Zygote is a test-only
dependency), `test/financialfft_example.jl` (the calibration page of FinancialFFT run verbatim, Heston by hand-written
Levenberg–Marquardt and H⁻¹Jᵀ, against `calibrate` and the ForwardDiff Jacobian of `calibrated_parameters` with the same
`Parametrization`: parameters 1e-8, sensitivities 1e-6; and the gradient of a Monte Carlo Asian option through the
calibration against the chain rule and recalibration with bumps of 1e-6, larger ones cross the kinks of the variance
floor since the calibrated Heston violates Feller), `test/cboe.jl` (a synthetic CBOE chain: parsing, filters, parity, the surface, and the adapter with a maturity
below the threshold, #2), `test/local/cboe_spx.jl` (local only, see above), `test/context.jl` (every file a CONTEXT.md lists exists, every source file and reference is reached from a contract, every contract is routed from the root CONTEXT.md), `test/aqua.jl` (Aqua.test_all), `test/gpu.jl` (run only if
`CUDA.functional()`, includes `calibrated_parameters` on device quotes). The tests define their own
`smile_prices` helper on top of `pricer`. The testsets report only at the end, so `runtests.jl` logs `Running`/`Finished … in N s` for the core testset and each included file (`logged_include`): a hung file shows up by name. A full local run with a GPU takes about 20 minutes (`models_methods.jl` 9, `gpu.jl` 6). Every FinancialFFT method runs on the GPU (FinancialFFT ≥ 0.5). The compat is FinancialFFT 0.6.1 and FinancialMonteCarlo 0.8, 0.9 (#30, #31): `ShiftedLogNormalMixture` became `ShiftedProcess(LogNormalMixture(η, λ, und), α)` with tuples, whose Monte Carlo is the Brigo–Mercurio local volatility (European prices converge to `AnalyticMethod`, tested with Sobol), and constructors throw `DomainError`.
