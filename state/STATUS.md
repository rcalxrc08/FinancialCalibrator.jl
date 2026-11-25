# Status

_What is in flight. Layer 4: volatile, rewritten freely; GitHub issues are the source of truth, this file is the index
an agent reads first._ Updated: 2026-11-25

## Current focus

0.3.0 on branch `issue-fmc72-underlying` (local, not pushed nor registered): FinancialMonteCarlo 0.10 and FinancialFFT 0.8,
the `Underlying` is an argument next to the model in `calibrate`, `calibrated_parameters`, `implicit_calibration`, `loss`,
`vol_errors` and the analytic pricer (FinancialMonteCarlo#72, D-007). Waits for the author's go to push, merge and register.

AD backend combinations that fail in the benchmark (#41). Fixed: groups 1, 3a, 3b, 4 (FinancialFFT 0.6.3, c1e256c); 3c on branch `issue-41-3c`.
The status table is #41 comment 7059648077 (edited in place); open points and next steps are comment 7061673022.

## Active work

| Item | Branch | Issue | Next step |
|:--|:--|:--|:--|
| Mooncake on the device, closed form: rule through `stateless_prices` (CUDA+Mooncake extension) | `issue-41-3c` | #41 group 3c | review; full suite, then merge; update the #41 comments |
| Mooncake Jacobian on the device | `master` | #41 group 3d | upstream: chalk-lab/Mooncake.jl#1485 (no `CuArray` method in its output check, fix verified); recheck on each Mooncake release |
| Enzyme on the device (forward and reverse) | `master` | #41 group 5 | same assertion and warnings as Enzyme#4094 in both modes: comment there with the reverse case |
| Enzyme `parmsRemoved` | `master` | #41 group 2 | toy reproducer for Enzyme |
| Groups 6 and 7 | `master` | #41 | 7 waits for Enzyme#4108 (primal wrong too, comment 7167307470); 6: recheck once #4108 is fixed, meanwhile reduce it (batched reverse over a `ComplexF32` Cos sum) |
| Docs and Mooncake as inner `x` timing | `master` | #41 group 8 | re-time the Heston Enzyme `calibrate` cells (still ✗ in `docs/src/backends.md`), add a test, full bench rerun |

## Known issues

| Issue | Symptom | Workaround |
|:--|:--|:--|
| #41 group 7 | Enzyme in Float32 on Heston/Fourier: wrong primal inside the differentiated loss (EnzymeAD/Enzyme.jl#4108), `Inf` and a zero gradient | ForwardDiff, or Float64 |
| #27 | shifted lognormal mixture on DAX converges to a degenerate corner (α ≈ −234) | — (optimum is degenerate, cond ≈ 5e9) |

## Technical debt and enhancements

- #44 SLNM loss allocations: parametrization done; next a scalar-loop loss for `AnalyticMethod` on the host (EnzymeR 147 → 110 μs, Mooncake 481 → 120 μs measured).
- #37 linear convergence of the polish; #32 Halley steps.
- #33 derivative-based first stage instead of Nelder–Mead; #34 avoid the p×n matrix for one direction.
- #25 a strike grid per maturity; #19 move `AnalyticMethod` out;
  #9 maturities in parallel. #10 done (2026-11-25): the method sizes per model are measured at the start and at the optimum by
  `docs/benchmarks.jl` (settings table of `docs/src/backends.md`); COS on Kou waits for FinancialFFT#101 (the range grows to L / λ₊).

## Working notes

Notes of an active initiative go next to this file as `state/YYYY-MM-DD-<topic>.md`; when it ends, what stays true
moves to `references/` or `references/decisions.md` and the note is deleted. The AD study of 2026-11-19/20 is already
in `references/autodiff.md`, `docs/src/backends.md` and `bench/results_2026-11-20.txt`.
