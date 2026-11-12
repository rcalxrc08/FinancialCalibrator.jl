"""
    calibrate(model, pricing_method, mktData::MarketData, opt_method = NelderMead(), opt_options = Optim.Options(); parametrization = Parametrization(), autodiff, polish = true, polish_backend, min_vega = 1e-3, verbose = false)

Calibrate the parameters of `model` to the implied volatility surface in `mktData`. Each smile is priced with
`pricing_method`, any `FinancialMonteCarlo.AbstractMethod`: a FinancialFFT method such as `CarrMadanMethod`,
`AnalyticMethod()` for models with a closed form, a `MonteCarloConfiguration` or a VibratoMonteCarlo method. Its execution
mode moves to `CudaMode` when the strikes of `mktData` are on the GPU. Only out of the money options are priced: puts
below the forward, calls above it; and only the quotes whose Black–Scholes vega at the quoted volatility is at least
`min_vega * S0`, whose prices are well above the accuracy of a numerical method. `ModelVolatility` is `NaN` at the quotes
left out.

The loss (`FinancialCalibrator.loss`), half the sum of the squared volatility errors, is minimized with the Optim optimizer `opt_method` and
settings `opt_options` (iterations, time limit, tolerances, ...), starting from the current parameters of `model`, over
the variables of `parametrization` (see [`Parametrization`](@ref); by default the parameters of
`FinancialMonteCarlo.get_parameters`); a
gradient-based optimizer gets its gradients from `autodiff` (an ADTypes backend). With `polish = true` the result is then
refined by Levenberg–Marquardt on the volatility errors (their Jacobian by `polish_backend`, an ADTypes backend through
DifferentiationInterface), so that the calibration ends where
the gradient of the loss vanishes, not where the tolerances of `opt_method` stopped it: the calibrated parameters are then
a smooth function of the quotes, which the implicit function theorem can differentiate. Nothing is printed unless
`verbose = true`.
"""
function calibrate(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, opt_method::Optim.AbstractOptimizer = NelderMead(), opt_options::Optim.Options = Optim.Options(); min_vega::Real = 1e-3, kwargs...)
    # what does not depend on the parameters is built once, not at every evaluation of the loss
    return calibrate_cached(model, pricing_method, mktData, loss_cache(model, pricing_method, mktData; min_vega = min_vega), opt_method, opt_options; kwargs...)
end

function calibrate_cached(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache, opt_method::Optim.AbstractOptimizer, opt_options::Optim.Options; parametrization::Parametrization = Parametrization(), kwargs...)::CalibratedData{T} where {T}
    x = optimal_variables(model, pricing_method, mktData, cache, opt_method, opt_options, parametrization; kwargs...)
    OptimalParameters = parametrization.to_params(x)
    OptimalValue = loss(OptimalParameters, model, pricing_method, mktData, cache)
    calibrated_model = FinancialMonteCarlo.set_parameters!(model, OptimalParameters)
    ModelVolatility = fill(T(NaN), length(mktData.TimeToMaturity), length(mktData.Strikes))
    for i in eachindex(cache.quoted)
        isempty(cache.quoted[i]) && continue
        ModelVolatility[i, cache.quoted[i]] = Array(model_volatilities(calibrated_model, model.underlying.S0, mktData, cache, i))
    end
    return CalibratedData(OptimalParameters, OptimalValue, ModelVolatility)
end

# The minimizer, in the variables of the parametrization: the optimizer, then the Levenberg–Marquardt polish.
function optimal_variables(model, pricing_method, mktData::MarketData, cache, opt_method::Optim.AbstractOptimizer, opt_options::Optim.Options, parametrization::Parametrization; autodiff::ADTypes.AbstractADType = default_backend(), polish::Bool = true, polish_backend::ADTypes.AbstractADType = default_backend(), verbose::Bool = false)
    to_params = parametrization.to_params
    loss_function(x::AbstractVector{<:Real}) = loss(to_params(x), model, pricing_method, mktData, cache)
    result = optimize(loss_function, parametrization.from_params(FinancialMonteCarlo.get_parameters(model)), opt_method, opt_options; autodiff = autodiff)
    x = result.minimizer
    if polish
        x = levenberg_marquardt(y -> vol_errors(to_params(y), model, pricing_method, mktData, cache), x; backend = polish_backend)
    end
    if verbose
        println(result)
        @show loss_function(x)
    end
    return x
end

# Levenberg–Marquardt on the residuals `errors(x)`, from `x`: (JᵀJ + μ diag(JᵀJ)) δ = -Jᵀe, with μ divided after a step
# that decreases the sum of squares, which is accepted, and multiplied after one that does not (or whose parameters are
# invalid: `errors` throws), which is rejected.
# When J lives on the GPU, the products JᵀJ and Jᵀe are computed there and only the p × p system comes to the host: a
# p × p solve on the device is a cuSOLVER factorization whose launches cost more than the copy of p² + p numbers, and
# copying J itself costs n × p. Mean time of one step, NVIDIA MX150, 1000 repetitions (products on the device and solve
# on the host / everything on the host after copying J / everything on the device):
#   n × p = 108 × 6 (the DAX surface, Heston):  412 / 161 / 1489 μs
#   n × p = 1000 × 6:                           518 / 1174 / 2322 μs
#   n × p = 10000 × 8:                          787 / 1720 / 1459 μs
# Copying J wins only on small surfaces, by less than 0.3 ms per step; a step also prices the surface once (the trial
# point) and differentiates it, which takes from tens to hundreds of milliseconds, so the choice is the one that does not
# degrade with the number of quotes.
# Exceptions at `x` itself, where the optimizer stopped, propagate: they are not a matter of the step.
function levenberg_marquardt(errors, x::AbstractVector{T}; backend::ADTypes.AbstractADType = default_backend(), iterations::Integer = 100, μ::Real = 1e-2, xtol::Real = 1e-12) where {T <: Real}
    e = errors(x)
    for _ = 1:iterations
        J = DI.jacobian(errors, backend, x)
        A = Array(transpose(J) * J)
        g = Array(transpose(J) * e)
        D = Diagonal(max.(diag(A), eps(T) * maximum(diag(A))))
        δ = -((A + μ * D) \ g)
        e_new = try
            errors(x + δ)
        catch err
            @debug "levenberg_marquardt: invalid step, rejected" x δ exception = err
            nothing
        end
        if e_new !== nothing && all(isfinite, e_new) && sum(abs2, e_new) < sum(abs2, e)
            x, e, μ = x + δ, e_new, μ / 3
            norm(δ) <= xtol * (1 + norm(x)) && break
        else
            μ *= 4
            # no step decreases the loss: the rounding floor of the volatilities
            μ > 1e12 && break
        end
    end
    return x
end
