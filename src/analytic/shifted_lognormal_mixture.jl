# The shifted lognormal mixture of Brigo and Mercurio: S_T = A_T + α F, with F = S0 e^{(r-d)T} the forward of S and A_T a
# mixture of lognormals with volatilities η, starting from A0 = (1 - α) S0. The model has n volatilities and n - 1
# weights λ: the weight of the last one is 1 - Σλ, so the mixture is a sum over n components of the same kind, and its
# arity is a type parameter of the kernel (a tuple), which a GPU thread can unroll.
analytic_price_type(model::ShiftedLogNormalMixture) = float(promote_type(typeof(model.underlying.S0), eltype(model.η), eltype(model.λ), typeof(model.α)))

function analytic_kernel(model::ShiftedLogNormalMixture, S0, r, d, T)
    α = model.α
    α < 1 || throw(DomainError(α, "the shift must be below 1, A0 = (1 - α) S0 is the spot of the mixture"))
    F = S0 * exp((r - d) * T)
    w = (model.λ..., 1 - sum(model.λ))
    return LognormalMixtureKernel((1 - α) * F, α * F, Tuple(model.η), w, r, T)
end
