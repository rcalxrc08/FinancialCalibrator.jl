# Decisions

_Architecture decisions, oldest first. Append; a reversed decision gets a new entry that says "supersedes D-NNN"._
(Layer 3 reference.) Read before a refactor: a structure that looks accidental may be an entry here. The evidence and
the details stay in the reference each entry points to.

## D-001 (2026-11-02, 7d788d0) Spot and dividends come from the model's `Underlying`

- **Decision:** `MarketData` holds maturities, rates, strikes and volatilities only; spot and the dividend curve are in
  the `Underlying` the model is built on, never passed as arguments.
- **Why:** one source of truth for the forward; a pricing method of FinancialMonteCarlo's interface already reads them
  from the model.
- **Consequences:** `adapter` returns `(underlying, mkt)`; rule in `CLAUDE.md`.
- **Status:** superseded in part by D-007 (the underlying is an argument next to the model; `MarketData` still holds no
  spot nor dividends)

## D-002 (date not recorded; ChainRulesCore made hard in #39) No package that differentiates is a hard dependency

- **Decision:** ForwardDiff, Zygote, Enzyme, Mooncake are weak dependencies or not dependencies at all;
  DifferentiationInterface/ADTypes (interfaces), ImplicitDifferentiation (author's choice) and ChainRulesCore (#39) are
  hard. `default_backend()` is ForwardDiff when loaded, FiniteDiff otherwise.
- **Why:** users choose the AD backend; loading a heavy AD package must not be the price of calibrating.
- **Consequences:** backend-specific rules live in `ext/`; details in `dependencies.md`.
- **Status:** active

## D-003 (2026-11-19, 966e245; #39) No branches on the differentiated path

- **Decision:** no `if`/ternary on a flag or a value in pricing and loss code: `ifelse`, `Bool * x`, `call_put_sign`
  weights, `max`/`min`; validations in `@non_differentiable` functions called directly (#43, which superseded the
  `ChainRulesCore.@ignore_derivatives` blocks of #39: their closures cost Zygote ~11 μs each). Branch-free part
  superseded by D-006.
- **Why:** Zygote is about 20x slower through branches, and cannot differentiate some of them at all. Zygote 0.7.19
  (2026-11-22) fixed the branch stacks (FluxML/Zygote.jl#1754): re-measured, a branch costs Zygote no more than a select
  (`autodiff.md`), so this reason holds only for older Zygote; the rule is relaxed in D-006.
- **Consequences:** rule in `CLAUDE.md`; details in `autodiff.md`.
- **Status:** active

## D-004 (2026-11-20, 872e4e1; #40) The `try` of `loss` stays outside what Zygote and Mooncake differentiate

- **Decision:** `loss_value` is the loss without the `try`; `loss` has a ChainRules `rrule` (Zygote) and a Mooncake
  `rrule!!` (extension) that run the `try` around the reverse pass, giving `Inf` and a zero gradient on a numerical
  failure.
- **Why:** neither Zygote nor Mooncake can differentiate a `catch`.
- **Consequences:** `loss`, its rrule and the Mooncake rule change together (`src/03_loss/CONTEXT.md`); #42 tracks that
  the zero gradient can hide a failing derivative.
- **Status:** active

## D-005 (2026-11-20, 633bba0 → 0aa5eff) Layered context (Interpretable Context Methodology)

- **Decision:** the context follows arXiv:2603.16021: `CLAUDE.md` (L0), `CONTEXT.md` routing (L1), stage contracts
  (L2), `references/` (L3), `state/` (L4, added 2026-11-21); `test/context.jl` keeps the map in sync with the code.
- **Why:** the previous single CLAUDE.md had grown to ~4700 words, loaded whole for every task.
- **Consequences:** a change updates the contracts, references and `state/STATUS.md` it touches in the same commit.
- **Status:** active

## D-006 (2026-11-22) Branches allowed on the differentiated path (supersedes the branch-free part of D-003)

- **Decision:** `if` and ternaries on flags or values are allowed in pricing and loss code; where a select is used,
  `Bool *` or weights rather than `ifelse` on a scalar flag; validations stay `@non_differentiable` functions (#43).
- **Why:** Zygote 0.7.19 fixed the branch stacks (FluxML/Zygote.jl#1754): on the toy kernel of #39 a ternary 18-21 →
  2.8-2.9 μs against 3.2-3.4 μs for `Bool *`, nested branches as fast as nested `ifelse` (`autodiff.md`); `ifelse` on a
  scalar flag stays ~4x `Bool *`. Mooncake and Enzyme never cared.
- **Consequences:** the test extras require Zygote 0.7.19; rule in `CLAUDE.md`; `autodiff.md`.
- **Status:** active

## D-007 (2026-11-25; FinancialMonteCarlo#72, 0.3.0) The underlying is an argument next to the model

- **Decision:** with FinancialMonteCarlo 0.10 the models hold only their parameters. `calibrate(model, und, method,
  mkt)`, `calibrated_parameters(model, und, method, mkt, vols)`, `loss_cache(model, und, method, mkt)`, and the public
  `vol_errors`/`loss` take the `Underlying` after the model; with a cache, the internal 5-argument forms read the spot
  and one `Underlying(S0, d_i)` per maturity (flat-equivalent dividend) from the cache.
- **Why:** one `set_parameters!` per evaluation and scalar dividends per maturity: the pricers integrate a number instead
  of the dividend curve at every loss and gradient (FinancialMonteCarlo#72, #44); the AD rules of the internal loss keep
  their arity.
- **Consequences:** breaking (0.3.0); `adapter` still returns `(underlying, mkt)`, now passed as is.
- **Status:** active
