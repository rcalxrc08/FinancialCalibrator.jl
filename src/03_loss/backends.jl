# The backend of the derivatives taken inside a calibration when none is given: ForwardDiff when it is loaded (exact
# derivatives, and a calibration has few parameters), finite differences otherwise. The packages that differentiate
# are weak dependencies; DifferentiationInterface, a dependency of Optim, loads its extension for ForwardDiff as soon
# as ForwardDiff is loaded, and FiniteDiff comes with Optim. `DI.check_available` answers by dispatch on the methods of
# that extension: no lock (unlike `Base.get_extension`), so a reverse mode backend can differentiate through the keyword
# defaults that call this function.
default_backend() = DI.check_available(ADTypes.AutoForwardDiff()) ? ADTypes.AutoForwardDiff() : ADTypes.AutoFiniteDiff()

# The backend of the polish (#36). ForwardDiff without a chunk size picks it from length(x), a run-time value: the type of
# the prepared Jacobian, and of everything the Levenberg–Marquardt loop computes from it, was not inferred (137 runtime
# dispatches reported by JET). The number of parameters of the model is a constant of its type
# (FinancialMonteCarlo.parameter_count), so the chunk is fixed from it, capped at ForwardDiff's default threshold (12;
# ForwardDiff is a weak dependency, so the constant is not read from it). A parametrization that changes the number of
# variables keeps the backend as given (a chunk larger than the input is an error).
function polish_backend_for(backend::ADTypes.AutoForwardDiff{nothing}, model, x::AbstractVector)
    N = FinancialMonteCarlo.parameter_count(typeof(model))
    return length(x) == N ? ADTypes.AutoForwardDiff{min(N, 12), typeof(backend.tag)}(backend.tag) : backend
end
polish_backend_for(backend::ADTypes.AbstractADType, model, x::AbstractVector) = backend
