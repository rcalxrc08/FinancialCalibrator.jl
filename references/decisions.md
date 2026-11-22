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
- **Status:** active

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
  `ChainRulesCore.@ignore_derivatives` blocks of #39: their closures cost Zygote ~11 μs each).
- **Why:** Zygote is about 20x slower through branches, and cannot differentiate some of them at all. Zygote 0.7.19
  (2026-11-22) fixed the branch stacks (FluxML/Zygote.jl#1754): re-measured, a branch costs Zygote no more than a select
  (`autodiff.md`), so this reason holds only for older Zygote; whether the rule stays is to decide.
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
