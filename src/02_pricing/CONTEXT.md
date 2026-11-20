# Stage 02: pricing

A model and a method price the smiles of one maturity.

## Inputs

| Load | Why |
|:--|:--|
| `pricer.jl` | `calibration_method(method, strikes)`: the method as given; the CUDA extension switches it to `CudaMode` for `CuArray` strikes |
| `analytic/analytic_method.jl` | `AnalyticMethod`, the generic closed-form `pricer`, `LognormalMixtureKernel` |
| `analytic/black_scholes.jl`, `analytic/shifted_lognormal_mixture.jl` | `analytic_price_type` and `analytic_kernel` per model |
| `references/pricing.md` | what each file defines |
| `references/autodiff.md` | the kernel must stay inferred under Zygote (no branches, no `oftype`, `@generated` sums) |

## Process

- There are no calibrator structs: a model/method pair is a `pricer` method. A new closed form is two methods,
  `analytic_price_type` and `analytic_kernel`, for its model.
- Closed forms come from FinancialToolbox (`blsprice`, `blkprice`), never written by hand.
- Generic in the number type; the kernel is an isbits callable, so a GPU broadcast takes it.

## Outputs

`pricer(model, ZeroRate(r), method, smile)` returns one price per strike, on the device of the strikes. Checked by
`test/runtests.jl` (closed form against Black–Scholes), `test/models_methods.jl` (models × methods), `test/gpu.jl`.
