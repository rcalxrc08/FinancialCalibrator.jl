# Calibrated parameters as a differentiable function of the quoted volatilities. The optimizer is not differentiated:
# at its solution x* the gradient of the loss vanishes, ∇ₓL(x*, v) = 0, and by the implicit function theorem
# ∂x*/∂v = -(∂ₓ∇ₓL)⁻¹ ∂ᵥ∇ₓL, from derivatives at x* only. ImplicitDifferentiation builds the rules from the solver and
# those conditions: a method for ForwardDiff duals, a ChainRules rrule (Zygote) and Enzyme rules; the extension for
# Mooncake adds a Mooncake rule on top of the ChainRules one.
struct CalibrationSolver{M, P, D, C, O, S, Z, A, B}
    model::M
    method::P
    mktData::D
    cache::C
    opt_method::O
    opt_options::S
    parametrization::Z
    autodiff::A
    polish_backend::B
end

function (s::CalibrationSolver)(vols::AbstractMatrix{<:Real})
    m = s.mktData
    quotes = MarketData(m.TimeToMaturity, m.MaturityDates, m.ZeroRates, m.Strikes, vols)
    # the quotes in the loss are those of the cache, chosen on mktData: the same for the solver and the conditions.
    # The polish brings the solution to where the conditions hold.
    x = optimal_variables(s.model, s.method, quotes, s.cache, s.opt_method, s.opt_options, s.parametrization; autodiff = s.autodiff, polish = true, polish_backend = s.polish_backend)
    return x, nothing
end

struct CalibrationConditions{M, P, D, C, Z, B}
    model::M
    method::P
    mktData::D
    cache::C
    parametrization::Z
    backend::B
end

# ∇ₓL(x, v) for L = ½‖vol_errors‖², in the variables x of the parametrization, by the backend of the conditions
# (ImplicitDifferentiation differentiates it again, with its own backends: nested differentiation)
function (c::CalibrationConditions)(vols::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, _)
    L(y) = sum(abs2, vol_errors(c.parametrization.to_params(y), c.model, c.method, c.mktData, c.cache; vols = vols)) / 2
    return DI.gradient(L, c.backend, x)
end

# The variables at the solution of the implicit function, for the quotes `vols`: a function of its own, so that the
# extension for Mooncake can give it a rule.
implicit_variables(implicit::ImplicitFunction, vols::AbstractMatrix{<:Real}) = first(implicit(vols))

"""
    calibrated_parameters(model, pricing_method, mktData, vols = mktData.Volatility, opt_method = NelderMead(), opt_options = Optim.Options(); parametrization = Parametrization(), autodiff, polish_backend, conditions_backend, implicit_backends = (x = ..., y = ...), min_vega = 1e-3)

The parameters that [`calibrate`](@ref) finds for the quoted implied volatilities `vols` (a matrix shaped as
`mktData.Volatility`, whose maturities, rates and strikes it takes; the quotes that enter the loss are chosen on
`mktData.Volatility`, see `calibrate`), as a function of `vols` that can be differentiated:

```julia
∂p_∂v = ForwardDiff.jacobian(V -> calibrated_parameters(model, method, mkt, V), mkt.Volatility)
```

The parameters are those of the model, `parametrization.to_params` of the variables the calibration searches over. The
derivatives come from the implicit function theorem at the solution, not from differentiating the optimizer: that needs
the gradient of the loss to vanish there, which the Levenberg–Marquardt polish of `calibrate` provides, and a smooth
loss around it.

Every automatic differentiation package can differentiate `calibrated_parameters` from outside: ForwardDiff through
its dual numbers, Zygote through the ChainRules rule of ImplicitDifferentiation, Enzyme through its Enzyme rules, and
Mooncake through a rule of FinancialCalibrator (an extension, loaded with Mooncake). Inside, the backends are ADTypes
objects, ForwardDiff when it is loaded and finite differences otherwise by default:
- `autodiff`, `polish_backend`: the gradients of the optimizer and the Jacobian of the polish, as in `calibrate`;
- `conditions_backend`: the gradient ``\\nabla_x L`` of the loss in the variables, the optimality conditions;
- `implicit_backends = (x = ..., y = ...)`: how ImplicitDifferentiation differentiates those conditions, with respect to
  the quotes (`x`) and to the variables (`y`). This differentiates the `conditions_backend` gradient again (nested).

ForwardDiff everywhere suits a model with few parameters. With a reverse mode outer backend and many quotes, `x` is the
one to change: the pullback needs ``(\\partial_v \\nabla_x L)^\\top w``, which forward mode gets one direction per quote.
"""
function calibrated_parameters(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, vols::AbstractMatrix{<:Real} = mktData.Volatility, opt_method::Optim.AbstractOptimizer = DEFAULT_OPTIMIZER, opt_options::Optim.Options = DEFAULT_OPTIONS; parametrization::Parametrization = Parametrization(), autodiff::ADTypes.AbstractADType = default_backend(), polish_backend::ADTypes.AbstractADType = default_backend(), conditions_backend::ADTypes.AbstractADType = default_backend(), implicit_backends::NamedTuple{(:x, :y)} = (x = default_backend(), y = default_backend()), min_vega::Real = 1e-3)
    size(vols) == size(mktData.Volatility) || throw(DimensionMismatch("vols is $(size(vols)), the surface of mktData $(size(mktData.Volatility))"))
    # the variables at the solution, with the rules of ImplicitDifferentiation; the parameters follow by the chain rule
    return parametrization.to_params(calibrated_variables(vols, model, pricing_method, mktData, opt_method, opt_options, parametrization, autodiff, polish_backend, conditions_backend, implicit_backends, min_vega))
end

# The implicit function built and called in one function, the quotes first and then the arguments of
# implicit_calibration: the extension for Enzyme gives it rules that build the implicit function outside of the
# differentiated code (Enzyme fails on a construction it traces, even marked inactive, #21), then apply the rules of
# ImplicitDifferentiation. Mooncake traces it, with its rules for implicit_calibration and implicit_variables.
calibrated_variables(vols::AbstractMatrix{<:Real}, setup::Vararg{Any, N}) where {N} = implicit_variables(implicit_calibration(setup...), vols)

# The implicit function of a calibration: what does not depend on the quotes being differentiated (the quotes in the
# loss, the solver, the conditions), in a function of its own with positional arguments, which the extension for
# Mooncake gives a zero derivative: Mooncake would otherwise differentiate the setup, copies from the GPU included.
function implicit_calibration(model, pricing_method, mktData, opt_method, opt_options, parametrization, autodiff, polish_backend, conditions_backend, implicit_backends, min_vega)
    cache = loss_cache(model, pricing_method, mktData; min_vega = min_vega)
    solver = CalibrationSolver(model, pricing_method, mktData, cache, opt_method, opt_options, parametrization, autodiff, polish_backend)
    conditions = CalibrationConditions(model, pricing_method, mktData, cache, parametrization, conditions_backend)
    # not strict: the cotangents of a reverse pass can be lazy arrays (Zygote gives a Fill-like vector for sum), of another
    # type than the one the pullback was prepared with
    return ImplicitFunction(solver, conditions; representation = MatrixRepresentation(), linear_solver = DirectLinearSolver(), backends = implicit_backends, strict = Val(false))
end
