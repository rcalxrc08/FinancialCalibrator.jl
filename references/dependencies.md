# Dependencies and the AD packages

_Why each dependency is (or is not) a hard one, and how the default backend is chosen._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

Dependencies, by design: **no package that differentiates is a hard dependency**. ADTypes and DifferentiationInterface
are interfaces (and hard dependencies of Optim through NLSolversBase anyway); ImplicitDifferentiation (the rules of
`calibrated_parameters`) is a hard dependency by the author's choice. ChainRulesCore is a hard dependency since #39 (an
interface, already loaded by FinancialMonteCarlo and ImplicitDifferentiation): the validations on the differentiated
path are in `ChainRulesCore.@ignore_derivatives`. ForwardDiff, Zygote, Enzyme, Mooncake are not:
`default_backend()` (`src/03_loss/backends.jl`) is `AutoForwardDiff()` when DifferentiationInterface's ForwardDiff extension is
loaded (i.e. the user loaded ForwardDiff) and `AutoFiniteDiff()` otherwise (FiniteDiff comes with NLSolversBase); every
backend keyword defaults to it (`DI.check_available`, not `Base.get_extension`: the latter takes a lock that Zygote
cannot differentiate through the keyword defaults). `ext/FinancialCalibratorMooncakeExt.jl` (triggered by Mooncake) gives Mooncake a
rule for `implicit_variables`: ImplicitDifferentiation's ChainRules rrule needs a `RuleConfig`, which Mooncake's
`@from_rrule` does not pass, so the extension calls it with a config without modes (the backends are always explicit in
`calibrated_parameters`, the config is never used) and moves the cotangent of the quotes itself. The setup of the implicit function is `implicit_calibration`, marked
`@zero_derivative` there: otherwise Mooncake traces it and fails on the device-to-host copies of the cache on the GPU.
AD support (CPU and GPU timings, failures) is the table in `docs/src/backends.md`: ForwardDiff, Zygote, Mooncake work
(Mooncake also on the GPU); Mooncake as inner `x` fails on the GPU (dual numbers broadcast over a CuArray); Enzyme
works in both modes on the host and in reverse mode on GPU quotes through `ext/FinancialCalibratorEnzymeExt.jl` (see
Gotchas, #20, #21). Removed as unused: DelimitedFiles (with the dead `web_retriever` and
`WebFileInput`, #15), FFTW, Random, FinancialFFT.
