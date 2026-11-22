# Status

_What is in flight. Layer 4: volatile, rewritten freely; GitHub issues are the source of truth, this file is the index
an agent reads first._ Updated: 2026-11-22

## Current focus

AD backend combinations that fail in the benchmark (#41). Fixed: groups 1, 3a, 3b, 4 (FinancialFFT 0.6.3, c1e256c); 3c on branch `issue-41-3c`.
The status table is #41 comment 7059648077 (edited in place); open points and next steps are comment 7061673022.

## Active work

| Item | Branch | Issue | Next step |
|:--|:--|:--|:--|
| Mooncake on the device, closed form: rule through `stateless_prices` (CUDA+Mooncake extension) | `issue-41-3c` | #41 group 3c | review; full suite, then merge; update the #41 comments |
| Mooncake Jacobian on the device | `master` | #41 group 3d | upstream: chalk-lab/Mooncake.jl#1485 (no `CuArray` method in its output check, fix verified); recheck on each Mooncake release |
| Enzyme on the device (forward and reverse) | `master` | #41 group 5 | same assertion and warnings as Enzyme#4094 in both modes: comment there with the reverse case |
| Enzyme `parmsRemoved` | `master` | #41 group 2 | toy reproducer for Enzyme |
| Groups 6 and 7 | `master` | #41, #42 | 7 waits for Enzyme#4108; 6: recheck once #4108 is fixed, meanwhile reduce it (batched reverse over a `ComplexF32` Cos sum) |
| Docs and Mooncake as inner `x` timing | `master` | #41 group 8 | re-time the Heston Enzyme `calibrate` cells (still ✗ in `docs/src/backends.md`), add a test, full bench rerun |

## Known issues

| Issue | Symptom | Workaround |
|:--|:--|:--|
| #42 | the `try` of `loss` turns a failing derivative into a zero gradient | compare with ForwardDiff when a backend gives a zero gradient |
| #38 | an unpriceable start point: `Inf` without the polish, `DomainError` with it | start from a priceable point |
| #27 | shifted lognormal mixture on DAX converges to a degenerate corner (α ≈ −234) | — (optimum is degenerate, cond ≈ 5e9) |

## Technical debt and enhancements

- #36 the polish loop is not inferred (23 runtime dispatches); #37 linear convergence of the polish; #32 Halley steps.
- #33 derivative-based first stage instead of Nelder–Mead; #34 avoid the p×n matrix for one direction.
- #25 a strike grid per maturity; #19 move `AnalyticMethod` out;
  #10 cheaper default method settings; #9 maturities in parallel.

## Working notes

Notes of an active initiative go next to this file as `state/YYYY-MM-DD-<topic>.md`; when it ends, what stays true
moves to `references/` or `references/decisions.md` and the note is deleted. The AD study of 2026-11-19/20 is already
in `references/autodiff.md`, `docs/src/backends.md` and `bench/results_2026-11-20.txt`.
