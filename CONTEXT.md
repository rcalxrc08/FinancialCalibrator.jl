# CONTEXT.md: task routing (Layer 1)

Find the task, open the contract it points to, and load only the inputs listed there.

| Task | Contract |
|:--|:--|
| New data source, a change in parsing, dates, the CBOE chain, `MarketData` | `src/01_market_data/CONTEXT.md` |
| A closed form for a model, the `pricer` hook, the GPU mode of a method | `src/02_pricing/CONTEXT.md` |
| Which quotes enter the loss, `vol_errors`, `loss`, a parametrization, the default backend | `src/03_loss/CONTEXT.md` |
| The optimizer, the polish, `calibrate`, `CalibratedData` | `src/04_calibration/CONTEXT.md` |
| `calibrated_parameters`, implicit derivatives, AD backends inside them | `src/05_sensitivities/CONTEXT.md` |
| Rules for Enzyme, Mooncake, CUDA | `ext/CONTEXT.md` |
| Adding or running tests, CI | `test/CONTEXT.md` |
| Documentation pages, figures, benchmarks | `docs/CONTEXT.md` |
| The AD benchmark of every backend combination | `bench/CONTEXT.md` |
| The logo and the favicon | `etc/CONTEXT.md` |
| Releases, compat, formatting | `references/workflows.md` |

Shared references (Layer 3), each loaded only when a contract lists it: `public_api.md` (the user-facing functions and
data flow), `conventions.md`, `autodiff.md`, `dependencies.md`, `workflows.md`, `testing.md`, and one per stage:
`market_data.md`, `pricing.md`, `loss.md`, `optimization.md`, `sensitivities.md`; `docs_design.md` for the site.

The stages communicate through Julia values, not files: 01 returns `(underlying, mkt::MarketData)`; 02 gives
`pricer(model, ZeroRate(r), method, smile)`; 03 builds `loss_cache` and evaluates `vol_errors`/`loss`; 04 returns
`CalibratedData`; 05 returns the parameters as a differentiable function of `mkt.Volatility`.
