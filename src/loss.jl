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
    # set_parameters! returns a new model, it does not mutate `model`. Its type is not inferred (it rebuilds the model
    # field by field): the loop over the maturities is a function of its own, compiled for the new model.
    new_model = FinancialMonteCarlo.set_parameters!(model, params)
    return maturity_errors(new_model, model.underlying.S0, mktData, cache, vols)
end

function maturity_errors(new_model, S0, mktData::MarketData, cache, vols)
    errors = map(filter(i -> !isempty(cache.quoted[i]), eachindex(cache.quoted))) do i
        return model_volatilities(new_model, S0, mktData, cache, i) .- vols[i, cache.quoted[i]]
    end
    return reduce(vcat, errors)
end

# The model implied volatilities of the quotes of maturity i that enter the loss.
function model_volatilities(new_model, S0, mktData::MarketData, cache, i::Integer)
    prices = vec(pricer(new_model, cache.zero_rates[i], cache.method, cache.smiles[i]))
    minimum(prices) > 0 || throw(DomainError(minimum(prices), "a model price is not positive"))
    return blsimpv.(S0, cache.strikes[i], mktData.ZeroRates[i], mktData.TimeToMaturity[i], prices, cache.dividends[i], cache.is_call[i])
end

"""
    loss(params, model, method, mktData, cache = loss_cache(model, method, mktData))

Half the sum of the squared volatility errors of [`vol_errors`](@ref): smooth in the parameters, also where a maturity
is fitted exactly. Invalid parameters (an exception while pricing) are penalised with a large constant.
"""
loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache = loss_cache(model, method, mktData)) where {T} = loss(promote_type(eltype(params), T), params, model, method, mktData, cache)

# the type of the result as a parameter: the sum of the errors is not inferred (see vol_errors), its conversion is
function loss(::Type{R}, params, model, method, mktData, cache)::R where {R}
    try
        return R(sum(abs2, vol_errors(params, model, method, mktData, cache)) / 2)
    catch e
        # invalid parameters are penalised; enable with JULIA_DEBUG=FinancialCalibrator to see why
        @debug "loss: exception, penalised" params exception = e
        return R(10000)
    end
end
