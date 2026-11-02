"""
    calibrate(model, pricing_method, mktData::MarketData, opt_method = NelderMead(), opt_options = Optim.Options(); verbose = false)

Calibrate the parameters of `model` to the implied volatility surface in `mktData`. Each smile is priced with
`pricing_method` (a FinancialFFT method such as `CarrMadanMethod`, or `ClosedFormMethod()` for models with a closed
form) and the loss is minimized with the Optim optimizer `opt_method` and settings `opt_options` (iterations, time limit,
tolerances, ...), starting from the current parameters of `model`. Nothing is printed unless `verbose = true`.
"""
function calibrate(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, opt_method::Optim.AbstractOptimizer = NelderMead(), opt_options::Optim.Options = Optim.Options(); verbose::Bool = false)::CalibratedData{T} where {T}
    LossFunction(v::AbstractVector{<:Real}) = Loss(v, model, pricing_method, mktData)
    InitialPoint = FinancialMonteCarlo.get_parameters(model)
    result = optimize(LossFunction, InitialPoint, opt_method, opt_options)
    OptimalParameters = result.minimizer
    if verbose
        @show LossFunction(OptimalParameters)
        println(result)
    end
    ModelVolatility = Matrix{T}(undef, length(mktData.TimeToMaturity), length(mktData.Strikes))
    for i = 1:length(mktData.TimeToMaturity)
        OutputPriceVec = smile_prices(model, pricing_method, OptimalParameters, mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], mktData.ImpliedDividend[i])
        ModelVolatility[i, 1:end] = blsimpv.(mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, mktData.ImpliedDividend[i])
    end
    return CalibratedData(OptimalParameters, result.minimum, ModelVolatility)
end
