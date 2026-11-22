# The shifted lognormal mixture of Brigo and Mercurio: S_T = A_T + α F, with F = S0 e^{(r-d)T} the forward of S and A_T a
# mixture of lognormals with volatilities η, starting from A0 = (1 - α) S0. The model has n volatilities and n - 1
# weights λ: the weight of the last one is 1 - Σλ, so the mixture is a sum over n components of the same kind, and its
# arity is a type parameter of the kernel (a tuple), which a GPU thread can unroll.
# FinancialMonteCarlo writes it as ShiftedProcess(LogNormalMixture(η, λ, underlying), α) (η and λ tuples), and the
# displaced diffusion as ShiftedProcess(BlackScholesProcess(σ, underlying), α): a mixture of one component. Its Monte
# Carlo simulates the Brigo–Mercurio local volatility, whose European prices are these closed forms.
const LognormalMixtureModel = Union{BlackScholesProcess, LogNormalMixture}

mixture_volatilities(model::BlackScholesProcess) = (model.σ,)
mixture_volatilities(model::LogNormalMixture) = model.η
mixture_component_weights(model::BlackScholesProcess) = (one(model.σ),)
mixture_component_weights(model::LogNormalMixture) = (model.λ..., 1 - sum(model.λ))

analytic_price_type(model::LogNormalMixture) = float(promote_type(typeof(model.underlying.S0), eltype(model.η), eltype(model.λ)))
analytic_price_type(model::ShiftedProcess{<:LognormalMixtureModel}) = float(promote_type(analytic_price_type(model.model), typeof(model.α)))

function analytic_kernel(model::LogNormalMixture, S0, r, d, T)
    F = S0 * exp((r - d) * T)
    return LognormalMixtureKernel(F, zero(F), mixture_volatilities(model), mixture_component_weights(model), r, T)
end

# a validation, not a derivative (#39): @non_differentiable, called without an @ignore_derivatives closure (#43)
check_shift(α) = α < 1 ? nothing : throw_domain_error(α, "the shift must be below 1, A0 = (1 - α) S0 is the spot of the mixture")
ChainRulesCore.@non_differentiable check_shift(::Any)

function analytic_kernel(model::ShiftedProcess{<:LognormalMixtureModel}, S0, r, d, T)
    α = model.α
    check_shift(α)
    F = S0 * exp((r - d) * T)
    return LognormalMixtureKernel((1 - α) * F, α * F, mixture_volatilities(model.model), mixture_component_weights(model.model), r, T)
end
