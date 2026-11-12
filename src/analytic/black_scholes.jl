# Black and Scholes: a mixture of one lognormal, with no shift.
analytic_price_type(model::BlackScholesProcess) = float(promote_type(typeof(model.underlying.S0), typeof(model.σ)))

function analytic_kernel(model::BlackScholesProcess, S0, r, d, T)
    F = S0 * exp((r - d) * T)
    return LognormalMixtureKernel(F, zero(F), (model.σ,), (one(model.σ),), r, T)
end
