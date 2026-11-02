"""
    calibrate(model, pricing_method, mktData::MarketData, opt_method = NelderMead())

Calibrate the parameters of `model` to the implied volatility surface in `mktData`. Each smile is priced with
`pricing_method` (a FinancialFFT method such as `CarrMadanMethod`, or `ClosedFormMethod()` for models with a closed
form) and the loss is minimized with the Optim optimizer `opt_method`, starting from the current parameters of `model`.
"""
function calibrate(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, opt_method::Optim.AbstractOptimizer = NelderMead())::CalibratedData
    LossFunction(v::Array{Float64}) = Loss(v, model, pricing_method, mktData)
    InitialPoint = FinancialMonteCarlo.get_parameters(model)
    result = optimize(LossFunction, InitialPoint, opt_method)
    OptimalParameters = result.minimizer
    @show LossFunction(OptimalParameters)
    println("RESULT")
    println(result)
    println(summary(result))
    ModelVolatility = Matrix{Float64}(undef, length(mktData.TimeToMaturity), length(mktData.Strikes))
    for i = 1:length(mktData.TimeToMaturity)
        OutputPriceVec = smile_prices(model, pricing_method, OptimalParameters, mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], mktData.ImpliedDividend[i])
        ModelVolatility[i, 1:end] = blsimpv.(mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, mktData.ImpliedDividend[i])
    end
    return CalibratedData(OptimalParameters, result.minimum, ModelVolatility)
end
