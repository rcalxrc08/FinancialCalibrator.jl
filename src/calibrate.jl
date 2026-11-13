# The default optimizer and options, built once: a default argument is evaluated in the body of the method that a
# reverse mode backend traces, and NelderMead() builds a Dict of its keywords, whose constructor has a try/catch on
# Julia 1.10, which Mooncake cannot differentiate. Both are immutable.
const DEFAULT_OPTIMIZER = NelderMead()
# The default options hand over to the polish: the optimizer stops once its simplex (or gradient) is below 1e-6, and the
# Levenberg–Marquardt polish converges from there in a few steps. Optim's default, 1e-8, made the optimizer crawl to a
# point the polish would reach anyway: on DAX the same optimum for the seven models tried, Kou 2.7 s -> 1.4 s, Merton
# 0.9 s -> 0.6 s (#22). Without the polish the default is Optim's own (FULL_OPTIONS).
const DEFAULT_OPTIONS = Optim.Options(g_tol = 1e-6)
const FULL_OPTIONS = Optim.Options()

"""
    calibrate(model, pricing_method, mktData::MarketData, opt_method = NelderMead(), opt_options = Optim.Options(g_tol = 1e-6); parametrization = default_parametrization(model), autodiff, polish = true, polish_backend, min_vega = 1e-3, verbose = false)

Calibrate the parameters of `model` to the implied volatility surface in `mktData`. Each smile is priced with
`pricing_method`, any `FinancialMonteCarlo.AbstractMethod`: a FinancialFFT method such as `CarrMadanMethod`,
`AnalyticMethod()` for models with a closed form, a `MonteCarloConfiguration` or a VibratoMonteCarlo method. Its execution
mode moves to `CudaMode` when the strikes of `mktData` are on the GPU. Only out of the money options are priced: puts
below the forward, calls above it; and only the quotes whose Black–Scholes vega at the quoted volatility is at least
`min_vega * S0`, whose prices are well above the accuracy of a numerical method. `ModelVolatility` is `NaN` at the quotes
left out.

The loss (`FinancialCalibrator.loss`), half the sum of the squared volatility errors, is minimized with the Optim optimizer `opt_method` and
settings `opt_options` (iterations, time limit, tolerances, ...), starting from the current parameters of `model`, over
the variables of `parametrization` (see [`Parametrization`](@ref); by default [`default_parametrization`](@ref), on
which every point is a valid model). The default options stop the optimizer at a tolerance of 1e-6 and leave the rest to
the polish; with `polish = false` the default is Optim's own `Optim.Options()`. To start from a previous solution, pass
the model with those parameters (`FinancialMonteCarlo.set_parameters!(model, p)`); a
gradient-based optimizer gets its gradients from `autodiff` (an ADTypes backend). With `polish = true` the result is then
refined by Levenberg–Marquardt on the volatility errors (their Jacobian by `polish_backend`, an ADTypes backend through
DifferentiationInterface), so that the calibration ends where
the gradient of the loss vanishes, not where the tolerances of `opt_method` stopped it: the calibrated parameters are then
a smooth function of the quotes, which the implicit function theorem can differentiate. Nothing is printed unless
`verbose = true`.
"""
function calibrate(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, opt_method::Optim.AbstractOptimizer = DEFAULT_OPTIMIZER, opt_options::Optim.Options = DEFAULT_OPTIONS; min_vega::Real = 1e-3, kwargs...)
    # what does not depend on the parameters is built once, not at every evaluation of the loss
    return calibrate_cached(model, pricing_method, mktData, loss_cache(model, pricing_method, mktData; min_vega = min_vega), opt_method, opt_options; kwargs...)
end

function calibrate_cached(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache, opt_method::Optim.AbstractOptimizer, opt_options::Optim.Options; parametrization::Parametrization = default_parametrization(model), kwargs...)::CalibratedData{T} where {T}
    x, e = optimal_variables_and_errors(model, pricing_method, mktData, cache, opt_method, opt_options, parametrization; kwargs...)
    OptimalParameters = parametrization.to_params(x)
    ModelVolatility = fill(T(NaN), length(mktData.TimeToMaturity), length(mktData.Strikes))
    e === nothing && return CalibratedData(OptimalParameters, T(Inf), ModelVolatility)
    # the value and the model volatilities from the errors at the optimum, which the polish has already computed: the
    # surface is not priced again (#22). The errors are maturity by maturity, in the order of the quotes of the cache.
    e = Array(e)
    offset = 0
    for i in eachindex(cache.quoted)
        q = cache.quoted[i]
        isempty(q) && continue
        ModelVolatility[i, q] = Array(mktData.Volatility[i, q]) .+ e[(offset+1):(offset+length(q))]
        offset += length(q)
    end
    return CalibratedData(OptimalParameters, T(sum(abs2, e) / 2), ModelVolatility)
end

# The minimizer, in the variables of the parametrization: the optimizer, then the Levenberg–Marquardt polish.
# Inferred (the result of optimize and the polish): Enzyme's rules for the implicit function need the solver of
# calibrated_parameters inferred, they check the type of the shadow against the type of the result.
optimal_variables(args...; kwargs...) = first(optimal_variables_and_errors(args...; kwargs...))

# The minimizer and the volatility errors there (`nothing` if they cannot be priced), so that calibrate needs no other
# pricing of the surface. With the default options and no polish, Optim runs with its own tolerances (FULL_OPTIONS).
function optimal_variables_and_errors(model, pricing_method, mktData::MarketData, cache, opt_method::Optim.AbstractOptimizer, opt_options::Optim.Options, parametrization::Parametrization; autodiff::ADTypes.AbstractADType = default_backend(), polish::Bool = true, polish_backend::ADTypes.AbstractADType = default_backend(), verbose::Bool = false)
    to_params = parametrization.to_params
    loss_function(x::AbstractVector{<:Real}) = loss(to_params(x), model, pricing_method, mktData, cache)
    errors(x) = vol_errors(to_params(x), model, pricing_method, mktData, cache)
    x0 = parametrization.from_params(FinancialMonteCarlo.get_parameters(model))
    options = !polish && opt_options === DEFAULT_OPTIONS ? FULL_OPTIONS : opt_options
    result = optimize(loss_function, x0, opt_method, options; autodiff = autodiff)
    x = result.minimizer
    x, e = polish ? levenberg_marquardt_and_errors(errors, x; backend = polish_backend) : (x, numerical_or_nothing(errors, x))
    if verbose
        println(result)
        println("loss after the polish: ", e === nothing ? Inf : sum(abs2, e) / 2)
    end
    return x, e
end

# errors(x), or nothing where the parameters cannot be priced
function numerical_or_nothing(errors, x)
    try
        return errors(x)
    catch err
        numerical_failure(err) || rethrow()
        return nothing
    end
end

# J, JᵀJ and Jᵀe at x (e the residuals there): the products on the device of J, the p × p system on the host.
function normal_equations(errors, prep, backend, x, e)
    J = DI.jacobian(errors, prep, backend, x)
    return J, Array(transpose(J) * J), Array(transpose(J) * e)
end

# Whether a step from the residuals e to e_new is accepted, with J the Jacobian and g = Jᵀe at the current point, and
# the gradient phase: `nothing` until a step is accepted on the gradient, then the sum of squares where the phase began
# and the last gradient accepted (each norm of Jᵀe_new with the J of its own step).
# Returns (accepted, phase). Outside the phase a step is accepted if it decreases the sum of squares, computed as
# Σ (e_new - e)(e_new + e), without the cancellation of |e_new|² - |e|²; or else, starting the phase, if it leaves the
# sum within sqrt(eps) (relative) and halves the gradient (Jᵀe_new, with the same J). In the phase every step must halve
# the last gradient accepted and keep the sum within sqrt(eps) of where the phase began.
# Why: near the optimum the decrease a step promises is far below the noise of the residuals. On DAX with Black-Scholes
# and GPU quotes, -6e-18 against changes of ±1e-15 from the implied volatilities (inverted to ~1e-14): tested on the
# values alone, the polish stopped where the noise hid the decrease, 9e-10 (relative) short of the optimum, with a
# gradient of 2e-8, which is resolved far better (noise ~1e-13). Why the phase: a step accepted on the gradient may
# raise the sum a little, and the next one could then be accepted on a noisy "decrease"; with a FiniteDiff Jacobian the
# two alternated for all the iterations (a random walk, 10 times slower). In the phase the gradient only halves, so the
# polish stops after a few steps, at the noise of the gradient; halving the last gradient accepted, not g, which comes
# from a new Jacobian at every step (a FiniteDiff Jacobian made g jump, and the phase walked as well).
function step_accepted(e, e_new, J, g, phase)
    S = sum(abs2, e)
    tolerance = sqrt(eps(typeof(S)))
    gradient = norm(transpose(J) * e_new)
    if phase === nothing
        change = sum((e_new .- e) .* (e_new .+ e))
        change < 0 && return true, phase
        change <= tolerance * S && gradient < norm(g) / 2 && return true, (S = S, gradient = gradient)
        return false, phase
    end
    sum(abs2, e_new) <= (1 + tolerance) * phase.S && gradient < phase.gradient / 2 && return true, (S = phase.S, gradient = gradient)
    return false, phase
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
# Type stability: the result has the type of `x` (Enzyme's rules for calibrated_parameters need the solver inferred).
# It used to be `Any`: `DI.jacobian(errors, backend, x)` unprepared lets ForwardDiff choose its chunk from `length(x)`,
# a run-time value, so the chunk, the dual numbers and the Jacobian are not inferred; the step δ = -(A + μD) \ g
# inherited that, and `x = x + δ` made the variables `Any` from the first accepted step, so the result was `Any`.
levenberg_marquardt(errors, x::AbstractVector{<:Real}; kwargs...) = first(levenberg_marquardt_and_errors(errors, x; kwargs...))

# The polish and the residuals at its end point.
function levenberg_marquardt_and_errors(errors, x::AbstractVector{T}; backend::ADTypes.AbstractADType = default_backend(), iterations::Integer = 100, μ::Real = 1e-2, xtol::Real = 1e-12) where {T <: Real}
    e = errors(x)
    # The Jacobian is prepared once (not at every step), but its type is still not inferred, since the chunk depends
    # on length(x). So the trial point is written into an array of the type of x: the variables keep their type
    # whatever the type of the step. x and x_trial swap at each accepted step, hence the copy: the caller's array is
    # not overwritten.
    x = copy(x)
    prep = DI.prepare_jacobian(errors, backend, x)
    x_trial = similar(x)
    # the Jacobian and the normal equations at x, recomputed only when a step is accepted: a rejected step leaves x
    J, A, g = normal_equations(errors, prep, backend, x, e)
    phase = nothing   # the gradient phase, see step_accepted
    for _ = 1:iterations
        # a Jacobian that is not finite (an overflow in the derivatives of the pricer) gives no direction: stop here
        all(isfinite, A) && all(isfinite, g) || break
        D = Diagonal(max.(diag(A), eps(T) * maximum(diag(A))))
        δ = -((A + μ * D) \ g)
        x_trial .= x .+ δ
        e_new = try
            errors(x_trial)
        catch err
            numerical_failure(err) || rethrow()
            @debug "levenberg_marquardt: invalid step, rejected" x δ exception = err
            nothing
        end
        accepted, phase = e_new === nothing || !all(isfinite, e_new) ? (false, phase) : step_accepted(e, e_new, J, g, phase)
        if accepted
            x, x_trial, e, μ = x_trial, x, e_new, μ / 3
            norm(δ) <= xtol * (1 + norm(x)) && break
            J, A, g = normal_equations(errors, prep, backend, x, e)
        else
            # no step decreases the loss: the rounding floor of the volatilities. A larger μ only shortens the step, so
            # a rejected step already below the tolerance ends the polish (it took 25 more rejections to reach μ > 1e12)
            norm(δ) <= xtol * (1 + norm(x)) && break
            μ *= 4
            μ > 1e12 && break
        end
    end
    return x, e
end
