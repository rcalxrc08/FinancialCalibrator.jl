
function Loss(params::Array{Float64}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData)::Float64
    loss = 0.0
    try
        @simd for i = 1:length(mktData.TimeToMaturity)
            OutputPriceVec = smile_prices(model, method, params, mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], mktData.ImpliedDividend[i])
            if (minimum(OutputPriceVec) <= 0.0)
                @show "negative price"
                return 1000000
            end
            VolaTmp = blsimpv.(mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, mktData.ImpliedDividend[i])
            loss += norm(VolaTmp-mktData.Volatility[i, 1:end])
        end
    catch e
        @show e
        loss=1e+4
    end
    return loss
end
