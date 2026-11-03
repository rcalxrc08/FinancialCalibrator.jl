# Everything the loss needs that depends neither on the parameters nor on the optimizer iteration: the smiles, the rates
# and the dividends of each maturity, and the execution mode. Built once per calibration.
function loss_cache(model::FinancialMonteCarlo.BaseProcess, mktData::MarketData)
    # the call flags live where the strikes live
    isCall = fill!(similar(mktData.Strikes, Bool), true)
    maturities = mktData.TimeToMaturity
    return (smiles = [FinancialMonteCarlo.EuropeanOptionSmile(t, mktData.Strikes, isCall) for t in maturities], zero_rates = [ZeroRate(r) for r in mktData.ZeroRates], dividends = [implied_dividend(model.underlying, t) for t in maturities], mode = pricing_mode(mktData.Strikes))
end

function loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache = loss_cache(model, mktData)) where {T}
    total = zero(T)
    try
        # set_parameters! returns a new model, it does not mutate `model`
        new_model = FinancialMonteCarlo.set_parameters!(model, params)
        for i = 1:length(mktData.TimeToMaturity)
            prices = vec(pricer(new_model, cache.zero_rates[i], method, cache.smiles[i], cache.mode))
            if minimum(prices) <= 0
                @debug "loss: negative price, penalised" params
                return T(1000000)
            end
            vols = blsimpv.(model.underlying.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], prices, cache.dividends[i])
            total += T(norm(vols .- @view(mktData.Volatility[i, :])))
        end
    catch e
        # invalid parameters are penalised; enable with JULIA_DEBUG=FinancialCalibrator to see why
        @debug "loss: exception, penalised" params exception = e
        total = T(10000)
    end
    return total
end
