# Mission

_Why the package exists and how to tell whether it succeeds._ (Layer 3 reference: stable, changes only with an entry in
`decisions.md`.)

## Purpose

Calibrate option pricing models (Lévy processes, Heston, shifted lognormal mixture, ...) to an implied volatility
surface, and make the calibrated parameters a differentiable function of the quotes, so that quote-to-risk derivatives
come from automatic differentiation (implicit function theorem) instead of bumping and recalibrating.

## Goals

- Any pricing method of FinancialMonteCarlo's `pricer(model, zero_rate, method, payoff)` interface calibrates with the
  same call: FinancialFFT's Fourier methods, FinancialMonteCarlo's Monte Carlo, VibratoMonteCarlo, the closed forms of
  `AnalyticMethod`.
- `calibrated_parameters(model, method, mkt, vols)` is differentiable with respect to `vols` with ForwardDiff, Zygote,
  Mooncake and Enzyme, on the host and on the GPU (status per combination: `docs/src/backends.md`, #41).
- Generic number types: Float32 works, on the host and on the device.
- Market data from files (`data/json/`) and from the CBOE.

## Constraints

- No package that differentiates is a hard dependency (`dependencies.md`, D-002 in `decisions.md`).
- Spot and dividends belong to the model's `Underlying` (D-001).
- No branches on the differentiated path (D-003).
- Julia ≥ 1.10 (`Project.toml`); CI on Julia 1.

## Success criteria

- `test/runtests.jl`: stationarity of `calibrate` on DAX, synthetic surfaces recovered by every model × method,
  derivatives of `calibrated_parameters` equal to −H⁻¹∂ᵥ∇L with every backend that is marked as working.
- `docs/src/backends.md` lists every backend combination with its timing or the issue that explains its failure.
- The Verify tables of the stage contracts pass in a fresh Kaimon session.

## Non-goals

- Pricing: models and methods live in FinancialMonteCarlo, FinancialFFT, VibratoMonteCarlo; this package only adds the
  closed forms of `AnalyticMethod` (to be moved to a package of their own, #19).
- Black–Scholes formulas and implied volatility: FinancialToolbox (`blsprice`, `blsimpv`).
- A general-purpose optimizer: Optim, plus the Levenberg–Marquardt polish.
