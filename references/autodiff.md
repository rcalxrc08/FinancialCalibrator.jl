# Automatic differentiation: backends, rules, traps

_Read before touching anything on the differentiated path._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

- Monte Carlo prices through FinancialMonteCarlo reseed at every call, so their AD gradient is the pathwise derivative
  on fixed paths: right for smooth payoffs, biased for barriers and binaries (the knock-out is not differentiated).
- Timings (VG/Lewis, 15 quotes, `docs/src/backends.md`): once compiled a gradient of `calibrated_parameters` costs about
  one calibration (0.1-0.2 s: NelderMead 128 loss evaluations at 0.75 ms, then the polish); the implicit derivatives ~8 ms.
  ForwardDiff's default chunk (8) re-runs the calibration per chunk: use one chunk for all the quotes. First calls after a
  calibration are compilation (ForwardDiff 14 s, Zygote 29 s, Mooncake 30 s, Enzyme 80-90 s; Mooncake as inner `x` is
  reverse over forward through the pricer: ~140 s, 4x slower at run time). On the MX150 the device is launch-bound on
  DAX (a loss 7-9 ms whatever the precision): Float32 pays only on large smiles (2000 strikes, Cos 4096: 6.7 ms vs 14 ms). Benchmark each backend in a fresh process: in one process later cases reuse compiled code.
  Enzyme (#20, #21): `calibrated_parameters` calls `calibrated_variables(vols, setup...)` (build the implicit function
  and call it); `ext/FinancialCalibratorEnzymeExt.jl` (weak dep EnzymeCore) gives it forward, augmented primal and
  reverse rules that build the implicit function as plain Julia code and delegate to ImplicitDifferentiation's own
  Enzyme rules with `Const(implicit)`, keeping only `y` of `(y, z)`. Enzyme must never trace the construction: even
  marked `inactive`, the struct went on its tape (a tape type mismatch on the host, an LLVM assertion in
  `createCacheForScope` on GPU quotes). Those rules check the type of the shadow against the result, so the solver must
  be inferred: `levenberg_marquardt` prepares its Jacobian once and writes the trial point into a copy of `x` (an
  unprepared `DI.jacobian` picks ForwardDiff's chunk at run time, which made everything `Any`); `@inferred` test of
  `optimal_variables` in `test/runtests.jl`, no type assertion. Tested in `test/models_methods.jl` and `test/gpu.jl`
  (reverse on the device compiles for about 18 minutes, then 2.3 s on BS/Cos DAX).
- Benchmark of every combination, 2026-11-20 (`bench/`, results in `bench/results_2026-11-20.txt`, tables in
  `docs/src/backends.md` "Which backend"): inside, Enzyme forward is the fastest (up to a third less than ForwardDiff on
  Lewis/Cos), reverse modes build the `vol_errors` Jacobian per quote (100-300x slower); outside, Zygote and Enzyme reverse
  tie with ForwardDiff on 99 quotes, ForwardDiff for the full Jacobian; the device pays from ~1000 quotes, Float32 only on
  the device (on the host slower than Float64). 43 failing combinations and one silently wrong gradient (Enzyme reverse,
  Heston loss in Float32, relative error 1) are explained in #41.
- **Branches on the differentiated path** (#39, relaxed by D-006): allowed since Zygote 0.7.19, which fixed their
  penalty (below); what follows is the history and the select forms that remain the better choice where a select is used.
  Before: no `if`/ternary on a flag or a value in the pricing and loss code. Use
  selects (`ifelse`, `Bool * x`, which has a strong zero, `call_put_sign` weights, `max`/`min`); validations that throw
  are named functions marked `ChainRulesCore.@non_differentiable` and called directly (`check_shift`, `α < 1` of the
  shifted mixture; `check_prices`; `check_surface_size`, the `size(vols)` checks), with the throw out of line (#43): an
  `@ignore_derivatives` block is a closure whose captures Zygote traces, ~11 μs each (FinancialMonteCarlo#80, #85).
  Measured 2026-11-22 (DAX, shifted mixture of 3, `AnalyticMethod`, Zygote gradient of `sum(abs2, vol_errors)`,
  minimum, fresh processes interleaved): 1838 / 2211 μs with the blocks, 1504 / 1616 μs without; value and
  ForwardDiff unchanged. Enzyme and Mooncake are indifferent to the pattern (FinancialMonteCarlo#85). Which backends differentiate `vol_errors` and `loss` (checked 2026-11-19, DAX, mixture with
  `AnalyticMethod` and VG with Lewis, against ForwardDiff, one process each): every backend both. Heston/Cos fails
  with Enzyme as `autodiff` of `calibrate` (#41, group 1). Zygote and Mooncake
  cannot differentiate a `catch` (Zygote "pop_exception" where it runs, Mooncake rejects the `try` in reverse mode), so
  `loss(::Type{R}, ...)` has rules that run the `try` themselves around the reverse pass of `loss_value` (the loss
  without it), `Inf` with a zero gradient on a `numerical_failure` (#40, fixed 2026-11-20): a ChainRules `rrule` with a
  `RuleConfig` (`rrule_via_ad`) for Zygote, and a Mooncake `rrule!!` in the extension (`@from_rrule` passes no config)
  around `build_rrule(loss_value, ...)`, which Mooncake caches per signature. The zero gradient is `zero(params)`: a
  `ZeroTangent` reaches DifferentiationInterface as `nothing`. Tested as `autodiff` of `calibrate` with `LBFGS()` on
  DAX in `test/models_methods.jl`. Zygote on `vol_errors` needed `analytic_prices` without `.=` into `similar`: one allocating broadcast through the
  isbits callable `PriceAs{PT}`, a closure over `PT` is not isbits and a GPU broadcast refuses it. On the device Zygote (0.7.18) differentiates the
  fields of a broadcast callable by rebuilding it with dual numbers: only fields typed by a bare, unbounded type parameter
  can hold them (`LognormalMixtureKernel` has one per field, its tuples included, and an outer constructor sets `N`), and
  `PriceAs` converts to `promote_type(PT, typeof(price))`, since `convert(PT, dual)` made the broadcast `Union{}` (#41
  group 4; before, the gradient was `nothing` with a warning). Mooncake refuses a callable with state on the device: the
  closed form there goes through a rule (`FinancialCalibratorCUDAMooncakeExt`) that differentiates `stateless_prices`,
  the same prices with a field-less `MixturePrice{PT, N}` and the numbers of the kernel as broadcast arguments; device
  gradients match ForwardDiff on the host to 1e-14 (#41 group 3c). Not the default for every backend: Zygote's device
  broadcast infers its duals for at most 10 arguments (`Zygote.dualize`, 0.7.18), and a mixture of N components takes
  2N + 6. Mooncake 0.5.72 also refuses a `CuArray` as the output of a pullback (`ValueAndPullbackReturnTypeError`, a
  `Ptr{CUstream_st}` inside the array), so the Mooncake Jacobian of `vol_errors` on the device fails (#41 group 3d): its output check has no method for
  `CuArray` and walks into the stream pointer of the array's memory (chalk-lab/Mooncake.jl#1485, with a fix verified
  to 3e-14; recheck on each Mooncake release). Mooncake 0.5.72 cannot differentiate `reduce(vcat, ...)` of `CuArray`s (the kernel
  launch of `_typed_vcat!` bitcasts a `CuPtr`, chalk-lab/Mooncake.jl#1484): the extension has a rule for it, with which
  the Fourier methods on the device match ForwardDiff (vg, heston: 4e-15, 3e-15, #41 group 3b). `loss_cache` is a
  zero-derivative function there (it depends on neither the parameters nor the differentiated quotes): built inside the
  differentiated loss on the device, it copies the vega mask to the host, whose CUDA copy has a `try` Mooncake rejects. ForwardDiff and
  Enzyme go through the `try` itself, Enzyme without runtime activity since FinancialMonteCarlo 0.9 (isbits `SVector` curves, FinancialMonteCarlo#79;
  with 0.8 any model built from active parameters on the `UnderlyingVec` of `adapter` needed `set_runtime_activity`). A
  closure annotated `Const` that captures the loss cache still makes Enzyme ask for runtime activity, on any underlying:
  pass model, market data and cache as `Const` arguments. Table in `docs/src/backends.md`. Zygote on `vol_errors` is
  correct but slow: with FinancialMonteCarlo 0.9 mixture 1.6 ms vs ForwardDiff 0.11 ms, BS/Lewis 16 vs 2.7 ms (0.8:
  7.7 ms on the mixture, of which `set_parameters!` ~0.2 ms and the dividend-curve `integral` ~0.45 ms per maturity;
  0.9 gives both reverse rules, FinancialMonteCarlo#80: 11-19 μs and 4 μs). What remains on the mixture is mostly the
  constructors' `@ignore_derivatives` closures (~11 μs each under Zygote 0.7.18, ~2.5 μs under 0.7.19, FinancialMonteCarlo#80,
  #85) and this package's
  pricing; with Lewis it is FinancialFFT's pricer. FinancialMonteCarlo's simulations are not for Zygote but its structs
  (models, curves, `set_parameters!`) must be. The branch stacks on that path (`Core._typeof_captured_variable`,
  which Zygote has a rule for only in forward mode, FluxML/Zygote.jl#1753, 33-57 of them; `Adapt.adapt_structure`;
  FinancialFFT's `pricer`) cost only 3-16% of it. Zygote upstream, filed 2026-11-19 (Zygote 0.7.18): #1751 `oftype`
  pullback not inferred (our `promote`; still open), #1752 `gradient`/`pullback` compiled for `f::Function` (4.4 μs ->
  57 ns specialized, the size-dependent slowdowns), #1753, #1754 branch-only stacks: #1752-#1754 fixed in Zygote 0.7.19
  (2026-11-22). Re-measured on 0.7.19 (fresh processes, the same toy kernel): ternary 18-21 → 2.8-2.9 μs, `Bool *`
  4.5-4.6 → 3.2-3.4 μs, `ifelse` on the flag 13.5-13.8 → 12.6 μs; two nested branches 30-43 → 5.4-5.7 μs against
  5.1-5.2 μs for nested `ifelse`. The Zygote penalty of branches is gone from 0.7.19 on (D-003). Measured 2026-11-19 (i7-8550U, Julia 1.13.4, DifferentiationInterface 0.7.22, one
  process per case, gradient in ns, ternary on `isCall` vs `Bool *`): toy kernel `exp(-rT) (F - κ)` Zygote 48100 vs
  6675 (7x), Mooncake 0.5.71 110 vs 106, Enzyme 0.13.221 (reverse) 66 vs 62; the mixture kernel (3 components, one
  option, from a vector) Zygote 43000 vs 42200, Mooncake 432 vs 421, Enzyme 797 vs 750. Mooncake and Enzyme compile the
  branch to straight code (differences within noise); Zygote stores the branch decisions in `Vector{Any}` stacks, so its
  pullback is not inferred (scalar toy with non-constant arguments: 6.9 μs and 52 allocations with the ternary, 22 ns
  with `Bool *` or `ifelse`; FluxML/Zygote.jl#1736 typed the stacks of loops but kept `Vector{Any}` for branch-only
  code on purpose). Benchmark Zygote with interpolated arguments and no splatting: literal arguments constant-fold the
  whole gradient (2.6 ns), and `@btime gradient($f, $(args)...)` splats at run time (`jl_f__apply_iterate`, boxed
  Float64s, microseconds that are the harness's). Gradients identical for calls and puts.
- **Zygote: one uninferred pullback makes the whole gradient dynamic.** Besides branches, two constructs did it in the
  mixture kernel (#39): `sum(map(closure, w, σ))` over tuples (7.5 μs vs 0.9 μs unrolled, pullback not inferred with the kernel's closure over
  `F`, `κ` and `blkprice`; a minimal tuple `map` is inferred, so the cause is not pinned down)
  and `oftype(x, y)` (Base's `oftype` is `y isa typeof(x) ? y : convert(...)`, a branch: FluxML/Zygote.jl#1751, Zygote
  0.7.18, still open in 0.7.19; recheck on new Zygote releases, `oftype` could come back then). The kernel now sums with the `@generated` `mixture_price` (nested `muladd`s, any arity) and ends
  with `ifelse(positive, promote(mixture, linear)...)`: Zygote gradient 18.2 μs -> 0.55 μs on a copy of the kernel,
  Mooncake ~0.75 μs and Enzyme ~1.05 μs either way. The real kernel built from the parameters inside the gradient (as
  `pricer` does), fresh process per arity: value / Zygote gradient 87/120, 179/262, 265/369, 345/462 ns for 1-4
  components (1.4x). With the kernel struct itself as the argument of `Zygote.gradient`, 2 and 4 components take
  2.5-3.3 μs (19 allocations, Zygote's accumulation of the tuple fields of a struct argument; odd arities are fast); no
  calibration differentiates that way.
  Also not inferred: an `if` on a compile-time constant (`Val`): Zygote transforms the untyped lowered code, so dispatch,
  not `if`, to choose code. A module-qualified call inside the differentiated function (`FC.mixture_price(...)` in a
  closure) makes Zygote differentiate `getproperty(::Module, ::Symbol)`, uninferred: 4.4 μs vs 0.36 μs (a benchmark
  artifact, the package calls its functions unqualified). Indexing a `Vector` (`x[i]`) costs a dense zero array per access in the pullback
  (~20 μs for 8 accesses): pass scalars, tuples or structs to differentiated kernels. Check with
  `Base.return_types(Zygote.pullback, Tuple{typeof(f), ...})`: no `Any` inside. Scripts were in the session scratchpad.
