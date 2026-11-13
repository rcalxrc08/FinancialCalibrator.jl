# Everything the loss needs that depends neither on the parameters nor on the optimizer iteration: the quotes that enter
# it, the smiles, the rates and the dividends of each maturity, and the pricing method in the mode of the strikes
# (calibration_method: CudaMode for a CuArray). Built once per calibration: the with_mode of a MonteCarloConfiguration
# rebuilds its random state.
# Each smile holds out of the money options only, puts below the forward and calls above it: an in the money option is
# mostly intrinsic value, and its implied volatility is ill-conditioned. For the same reason the quotes whose vega, at the
# quoted volatility, is below `min_vega * S0` are left out: their prices are so small that the error of a numerical
# method is a large error in volatility, and it moves erratically with the parameters (on the DAX surface one such quote,
# priced 1e-10, keeps the loss from being smooth at the 1e-6 level).
function loss_cache(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData; min_vega::Real = 1e-3)
    S0 = model.underlying.S0
    maturities, rates, strikes = mktData.TimeToMaturity, mktData.ZeroRates, mktData.Strikes
    dividends = [implied_dividend(model.underlying, t) for t in maturities]
    # the indices of the quotes kept, on the host; the strikes and the call flags live where the strikes live
    quoted = [findall(Array(blsvega.(S0, strikes, rates[i], maturities[i], @view(mktData.Volatility[i, :]), dividends[i]) .>= min_vega * S0)) for i in eachindex(maturities)]
    quoted_strikes = [strikes[q] for q in quoted]
    is_call = [k .>= S0 * exp((r - d) * t) for (k, t, r, d) in zip(quoted_strikes, maturities, rates, dividends)]
    smiles = [FinancialMonteCarlo.EuropeanOptionSmile(t, k, c) for (t, k, c) in zip(maturities, quoted_strikes, is_call)]
    return (quoted = quoted, strikes = quoted_strikes, smiles = smiles, is_call = is_call, zero_rates = [ZeroRate(r) for r in rates], dividends = dividends, method = calibration_method(method, strikes))
end

"""
    vol_errors(params, model, method, mktData, cache = loss_cache(model, method, mktData); vols = mktData.Volatility)

Model implied volatilities minus the quoted ones, for the parameters `params` of `model`: one entry per quote that enters
the loss (see `loss_cache`: out of the money options, vega at least `min_vega * S0`), maturity by maturity. Generic in the
number type of `params` and of `vols`, so that it can be differentiated with respect to both. Throws on invalid
parameters, and a `DomainError` on a price that is not positive.
"""
function vol_errors(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, cache = loss_cache(model, method, mktData); vols::AbstractMatrix{<:Real} = mktData.Volatility)
    # set_parameters! returns a new model, it does not mutate `model` (inferred since FinancialMonteCarlo 0.7; the loop
    # over the maturities stays a function of its own, a barrier should a model's type not be inferred).
    new_model = FinancialMonteCarlo.set_parameters!(model, params)
    return maturity_errors(new_model, model.underlying.S0, mktData, cache, vols)
end

# A loop with a concrete accumulator, not map and reduce: map cannot infer the element type of its closure with dual
# numbers, and the abstract vector it then returns makes the loss uninferred, which costs little in Julia but makes
# Mooncake compile a rule per dynamic call. A few maturities: the vcat of each step is cheap.
function maturity_errors(new_model, S0, mktData::MarketData, cache, vols)
    maturities = filter(i -> !isempty(cache.quoted[i]), eachindex(cache.quoted))
    errors = maturity_error(new_model, S0, mktData, cache, vols, first(maturities))
    for i in Iterators.drop(maturities, 1)
        errors = vcat(errors, maturity_error(new_model, S0, mktData, cache, vols, i))
    end
    return errors
end
maturity_error(new_model, S0, mktData::MarketData, cache, vols, i::Integer) = model_volatilities(new_model, S0, mktData, cache, i) .- vols[i, cache.quoted[i]]

# The model implied volatilities of the quotes of maturity i that enter the loss.
function model_volatilities(new_model, S0, mktData::MarketData, cache, i::Integer)
    prices = vec(pricer(new_model, cache.zero_rates[i], cache.method, cache.smiles[i]))
    minimum(prices) > 0 || throw(DomainError(minimum(prices), "a model price is not positive"))
    return blsimpv.(S0, cache.strikes[i], mktData.ZeroRates[i], mktData.TimeToMaturity[i], prices, cache.dividends[i], cache.is_call[i])
end

"""
    loss(params, model, method, mktData, cache = loss_cache(model, method, mktData))

Half the sum of the squared volatility errors of [`vol_errors`](@ref): smooth in the parameters, also where a maturity
is fitted exactly. `Inf` where the parameters cannot be priced (see `numerical_failure`), which an Optim optimizer ranks
as the worst point; any other exception propagates.
"""
loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache = loss_cache(model, method, mktData)) where {T} = loss(promote_type(eltype(params), T), params, model, method, mktData, cache)

# the type of the result as a parameter: the sum of the errors is not inferred (see vol_errors), its conversion is
function loss(::Type{R}, params, model, method, mktData, cache)::R where {R}
    try
        return R(sum(abs2, vol_errors(params, model, method, mktData, cache)) / 2)
    catch e
        numerical_failure(e) || rethrow()
        # enable with JULIA_DEBUG=FinancialCalibrator to see why
        @debug "loss: parameters that cannot be priced, Inf" params exception = e
        return R(Inf)
    end
end

# The exceptions that mean "these parameters cannot be priced", not a bug: a price that is not positive or has no implied
# volatility (DomainError), a model constructor rejecting its parameters (AssertionError: FinancialMonteCarlo checks them
# with @assert). The default parametrizations keep the constructors from ever failing; with Parametrization() they can.
# Anything else (a MethodError after an API change, ...) is a bug and propagates (#3). A constant penalty used to stand
# in for the loss there (#24): a plateau that a line search can stall on, and that looked like a value.
numerical_failure(e) = e isa Union{DomainError, AssertionError}
