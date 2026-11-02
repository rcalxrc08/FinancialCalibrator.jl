
function Loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}) where {T}
    loss = zero(T)
    try
        @simd for i = 1:length(mktData.TimeToMaturity)
            OutputPriceVec = smile_prices(model, method, params, mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], mktData.ImpliedDividend[i])
            if (minimum(OutputPriceVec) <= 0.0)
                @show "negative price"
                return T(1000000)
            end
            VolaTmp = blsimpv.(mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, mktData.ImpliedDividend[i])
            loss += T(norm(VolaTmp-mktData.Volatility[i, 1:end]))
        end
    catch e
        @show e
        loss = T(10000)
    end
    return loss
end
