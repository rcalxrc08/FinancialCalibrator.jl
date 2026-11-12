# The backend of the derivatives taken inside a calibration when none is given: ForwardDiff when it is loaded (exact
# derivatives, and a calibration has few parameters), finite differences otherwise. The packages that differentiate
# are weak dependencies; DifferentiationInterface, a dependency of Optim, loads its extension for ForwardDiff as soon
# as ForwardDiff is loaded, and FiniteDiff comes with Optim. `DI.check_available` answers by dispatch on the methods of
# that extension: no lock (unlike `Base.get_extension`), so a reverse mode backend can differentiate through the keyword
# defaults that call this function.
default_backend() = DI.check_available(ADTypes.AutoForwardDiff()) ? ADTypes.AutoForwardDiff() : ADTypes.AutoFiniteDiff()
