# Benchmark of the AD backends

## Inputs

| Load | Why |
|:--|:--|
| `bench.jl` | one case per process: scenario, arrays, part (A loss/Jacobian, B `calibrate`, C gradient of `calibrated_parameters`, J its Jacobian) and backends, from its header |
| `run_all.sh` | the list of cases, each in a fresh process with a 3000 s timeout; skips cases whose log already has a result |
| `results_2026-11-20.txt` | the last full run (laptop i7-8550U, NVIDIA MX150, CUDA 6.5.1, Enzyme 0.13.221, FinancialMonteCarlo 0.9) |
| `references/autodiff.md` | which backend differentiates what, and the known failures |

## Process

Run by hand, never in CI: `cd bench && julia --project=. -e 'using Pkg; Pkg.instantiate()' && ./run_all.sh` (Git Bash on
Windows). It takes about 16 hours. `results.txt` and `logs/` are ignored by git; copy a full run to
`results_<date>.txt`. Compare run times only: the first call is compilation. A case that fails is a line with
`FAIL`, explained in FinancialCalibrator.jl#41.

## Outputs

The tables of `docs/src/backends.md` ("Which backend") come from the results file; update both together.
