
function loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}) where {T}
    total = zero(T)
    try
        # set_parameters! returns a new model, it does not mutate `model`
        new_model = FinancialMonteCarlo.set_parameters!(model, params)
        # the call flags live where the strikes live
        isCall = fill!(similar(mktData.Strikes, Bool), true)
        mode = pricing_mode(mktData.Strikes)
        @simd for i = 1:length(mktData.TimeToMaturity)
            smile = FinancialMonteCarlo.EuropeanOptionSmile(mktData.TimeToMaturity[i], mktData.Strikes, isCall)
            OutputPriceVec = vec(pricer(new_model, ZeroRate(mktData.ZeroRates[i]), method, smile, mode))
            if (minimum(OutputPriceVec) <= 0.0)
                @show "negative price"
                return T(1000000)
            end
            VolaTmp = blsimpv.(model.underlying.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, implied_dividend(model.underlying, mktData.TimeToMaturity[i]))
            total += T(norm(VolaTmp-mktData.Volatility[i, 1:end]))
        end
    catch e
        @show e
        total = T(10000)
    end
    return total
end
