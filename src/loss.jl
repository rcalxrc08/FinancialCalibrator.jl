# Everything the loss needs that depends neither on the parameters nor on the optimizer iteration: the smiles, the rates
# and the dividends of each maturity, and the pricing method in the mode of the strikes (calibration_method: CudaMode for
# a CuArray). Built once per calibration: the with_mode of a MonteCarloConfiguration rebuilds its random state.
function loss_cache(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData)
    # the call flags live where the strikes live
    isCall = fill!(similar(mktData.Strikes, Bool), true)
    maturities = mktData.TimeToMaturity
    return (smiles = [FinancialMonteCarlo.EuropeanOptionSmile(t, mktData.Strikes, isCall) for t in maturities], zero_rates = [ZeroRate(r) for r in mktData.ZeroRates], dividends = [implied_dividend(model.underlying, t) for t in maturities], method = calibration_method(method, mktData.Strikes))
end

function loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache = loss_cache(model, method, mktData))::T where {T}
    try
        # set_parameters! returns a new model, it does not mutate `model`. Its type is not inferred (it rebuilds the model
        # field by field): the sum over the maturities is a function of its own, compiled for the new model.
        new_model = FinancialMonteCarlo.set_parameters!(model, params)
        return maturity_losses(new_model, model.underlying.S0, cache.method, mktData, cache, params)
    catch e
        # invalid parameters are penalised; enable with JULIA_DEBUG=FinancialCalibrator to see why
        @debug "loss: exception, penalised" params exception = e
        return T(10000)
    end
end

function maturity_losses(new_model, S0, method, mktData::MarketData{T}, cache, params) where {T}
    total = zero(T)
    for i = 1:length(mktData.TimeToMaturity)
        prices = vec(pricer(new_model, cache.zero_rates[i], method, cache.smiles[i]))
        if minimum(prices) <= 0
            @debug "loss: negative price, penalised" params
            return T(1000000)
        end
        vols = blsimpv.(S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], prices, cache.dividends[i])
        total += T(norm(vols .- @view(mktData.Volatility[i, :])))
    end
    return total
end
