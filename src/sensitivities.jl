# Calibrated parameters as a differentiable function of the quoted volatilities. The optimizer is not differentiated:
# at its solution p* the gradient of the loss vanishes, ∇ₚL(p*, v) = 0, and by the implicit function theorem
# ∂p*/∂v = -(∂ₚ∇ₚL)⁻¹ ∂ᵥ∇ₚL, from derivatives at p* only. ImplicitDifferentiation builds the rules from the solver and
# those conditions: a method for ForwardDiff duals and a ChainRules rrule (Zygote; Mooncake with @from_chainrules).

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

# ∇ₓL(x, v) for L = ½‖vol_errors‖², in the variables x of the parametrization, by the backend of the conditions (ImplicitDifferentiation differentiates it again, with
# its own backends: nested differentiation, ForwardDiff over ForwardDiff by default)
function (c::CalibrationConditions)(vols::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, _)
    L(y) = sum(abs2, vol_errors(c.parametrization.to_params(y), c.model, c.method, c.mktData, c.cache; vols = vols)) / 2
    return DI.gradient(L, c.backend, x)
end

"""
    calibrated_parameters(model, pricing_method, mktData, vols = mktData.Volatility, opt_method = NelderMead(), opt_options = Optim.Options(); parametrization = Parametrization(), autodiff = AutoForwardDiff(), polish_backend = AutoForwardDiff(), conditions_backend = AutoForwardDiff(), implicit_backends = (x = AutoForwardDiff(), y = AutoForwardDiff()), min_vega = 1e-3)

The parameters that [`calibrate`](@ref) finds for the quoted implied volatilities `vols` (a matrix shaped as
`mktData.Volatility`, whose maturities, rates and strikes it takes; the quotes that enter the loss are chosen on
`mktData.Volatility`, see `calibrate`), as a function of `vols` that can be differentiated:

```julia
∂p_∂v = ForwardDiff.jacobian(v -> calibrated_parameters(model, method, mkt, reshape(v, size(mkt.Volatility))), vec(mkt.Volatility))
```

or with a reverse mode backend through ChainRules (Zygote, Mooncake with `@from_chainrules`). The parameters are those of the
model, `parametrization.to_params` of the variables the calibration searches over. The derivatives come from
the implicit function theorem at the solution, not from differentiating the optimizer: that needs the gradient of the
loss to vanish there, which the Levenberg–Marquardt polish of `calibrate` provides, and a smooth loss around it.

Any backend can differentiate `calibrated_parameters` from outside. Inside, the backends are ADTypes objects:
- `autodiff`, `polish_backend`: the gradients of the optimizer and the Jacobian of the polish, as in `calibrate`;
- `conditions_backend`: the gradient ∇ₚL of the loss in the parameters, the optimality conditions;
- `implicit_backends = (x = ..., y = ...)`: how ImplicitDifferentiation differentiates those conditions, with respect to the
  quotes (`x`) and to the parameters (`y`). This differentiates the `conditions_backend` gradient again (nested).

ForwardDiff everywhere suits a model with few parameters. With a reverse mode outer backend and many quotes, `x` is the one
to change: the pullback needs (∂ᵥ∇ₚL)ᵀ w, which forward mode gets one direction per quote.
"""
function calibrated_parameters(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, vols::AbstractMatrix{<:Real} = mktData.Volatility, opt_method::Optim.AbstractOptimizer = NelderMead(), opt_options::Optim.Options = Optim.Options(); parametrization::Parametrization = Parametrization(), autodiff::ADTypes.AbstractADType = ADTypes.AutoForwardDiff(), polish_backend::ADTypes.AbstractADType = ADTypes.AutoForwardDiff(), conditions_backend::ADTypes.AbstractADType = ADTypes.AutoForwardDiff(), implicit_backends::NamedTuple{(:x, :y)} = (x = ADTypes.AutoForwardDiff(), y = ADTypes.AutoForwardDiff()), min_vega::Real = 1e-3)
    size(vols) == size(mktData.Volatility) || throw(DimensionMismatch("vols is $(size(vols)), the surface of mktData $(size(mktData.Volatility))"))
    cache = loss_cache(model, pricing_method, mktData; min_vega = min_vega)
    solver = CalibrationSolver(model, pricing_method, mktData, cache, opt_method, opt_options, parametrization, autodiff, polish_backend)
    conditions = CalibrationConditions(model, pricing_method, mktData, cache, parametrization, conditions_backend)
    # not strict: the cotangents of a reverse pass can be lazy arrays (Zygote gives a Fill-like vector for sum), of another
    # type than the one the pullback was prepared with
    implicit = ImplicitFunction(solver, conditions; representation = MatrixRepresentation(), linear_solver = DirectLinearSolver(), backends = implicit_backends, strict = Val(false))
    # the variables at the solution, with the rules of ImplicitDifferentiation; the parameters follow by the chain rule
    return parametrization.to_params(first(implicit(vols)))
end
