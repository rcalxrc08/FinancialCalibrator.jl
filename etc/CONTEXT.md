# Logo and favicon

## Inputs

| Load | Why |
|:--|:--|
| `logo.jl` | draws `logo.png` and `favicon.ico`; `docs/make.jl` copies them into `docs/src/assets/` |
| `references/docs_design.md` | the design shared with the documentation and the other packages' logos |

## Process

Run `julia etc/logo.jl` from the root, in an environment with Luxor, ForwardDiff, FinancialFFT, FinancialMonteCarlo
and FinancialToolbox (the header of the script); commit the images with the script.

## Outputs

`logo.png` (README and docs) and `favicon.ico` (docs).
