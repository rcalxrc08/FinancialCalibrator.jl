# CLAUDE.md

FinancialCalibrator.jl calibrates option pricing models (Lévy processes, Heston, shifted lognormal mixture, ...) to an
implied volatility surface, and differentiates the calibrated parameters with respect to the quotes (implicit function
theorem). Pricing is delegated to FinancialMonteCarlo's `pricer(model, zero_rate, method, payoff)` and to whatever
method package the user loads (FinancialFFT, VibratoMonteCarlo); the optimizer is Optim.

This workspace follows the Interpretable Context Methodology (layered context, arXiv:2603.16021): load only what the
task needs.

| Layer | Where | What |
|:--|:--|:--|
| 0 | this file | identity, map, rules that hold everywhere |
| 1 | `CONTEXT.md` | which stage or folder a task belongs to, shared references |
| 2 | `src/0N_*/CONTEXT.md`, `ext/`, `test/`, `docs/` | a contract per stage: inputs to read, process, outputs and tests |
| 3 | `references/*.md` | stable facts and measurements, loaded when a contract lists them |

Map: `src/FinancialCalibrator.jl` includes the stages in order, `01_market_data` → `02_pricing` → `03_loss` →
`04_calibration` → `05_sensitivities`; `ext/` holds the extensions (CUDA, Enzyme, Mooncake); `test/` the suite;
`docs/` the Documenter site; `data/json/` the sample market data (DAX 22 Aug 2016); `examples/`, `etc/` scripts.

Rules for every change (details in `references/conventions.md` and `references/autodiff.md`):

- Spot and dividends live in the model's `Underlying`, never in `MarketData` nor as arguments.
- Generic floating point type: no `::Float64` nor Float64 literals on the pricing path; `Float32` must work.
- No branches on the differentiated path: `ifelse`, `Bool *`, weights; validations in `ChainRulesCore.@ignore_derivatives`.
- No package that differentiates is a hard dependency (`references/dependencies.md`).
- Reuse the author's packages (FinancialMonteCarlo, FinancialToolbox's `blsprice`/`blsimpv`, FinancialFFT,
  PatternVectors for quadrature weights) and established libraries; `@muladd` for every `a * b + c`.
- Format with the repo's `.JuliaFormatter.toml`; update the CONTEXT.md and references touched by a change in the same
  commit; commit messages end with the session's `Co-Authored-By` line.
