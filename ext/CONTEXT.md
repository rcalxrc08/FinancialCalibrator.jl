# Extensions

## Inputs

| Load | Why |
|:--|:--|
| `FinancialCalibratorCUDAExt.jl` | `calibration_method` for `CuArray` strikes: the method in `CudaMode` |
| `FinancialCalibratorEnzymeExt.jl` | forward, augmented primal and reverse rules of `calibrated_variables` |
| `FinancialCalibratorMooncakeExt.jl` | the Mooncake rules of `implicit_variables` and of `loss` (its `try`, #40), `implicit_calibration` as zero derivative |
| `references/autodiff.md` | why each rule exists (#20, #21) |
| `references/dependencies.md` | why these packages are weak dependencies |

## Process

An extension only adds methods; the types and functions it needs live in `src/`. Enzyme must never trace the
construction of the implicit function.

## Outputs

Checked by `test/models_methods.jl` (Enzyme, Mooncake) and `test/gpu.jl` (only with `CUDA.functional()`).
