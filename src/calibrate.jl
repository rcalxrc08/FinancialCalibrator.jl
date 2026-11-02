"""
    calibrate(model, pricing_method, mktData::MarketData, opt_method = NelderMead(), opt_options = Optim.Options(); verbose = false)

Calibrate the parameters of `model` to the implied volatility surface in `mktData`. Each smile is priced with
`pricing_method` (a FinancialFFT method such as `CarrMadanMethod`, or `AnalyticMethod()` for models with a closed
form) and the loss is minimized with the Optim optimizer `opt_method` and settings `opt_options` (iterations, time limit,
tolerances, ...), starting from the current parameters of `model`. Nothing is printed unless `verbose = true`.
"""
function calibrate(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, opt_method::Optim.AbstractOptimizer = NelderMead(), opt_options::Optim.Options = Optim.Options(); verbose::Bool = false)::CalibratedData{T} where {T}
    loss_function(v::AbstractVector{<:Real}) = loss(v, model, pricing_method, mktData)
    InitialPoint = FinancialMonteCarlo.get_parameters(model)
    result = optimize(loss_function, InitialPoint, opt_method, opt_options)
    OptimalParameters = result.minimizer
    if verbose
        @show loss_function(OptimalParameters)
        println(result)
    end
    calibrated_model = FinancialMonteCarlo.set_parameters!(model, OptimalParameters)
    isCall = fill!(similar(mktData.Strikes, Bool), true)
    mode = pricing_mode(mktData.Strikes)
    ModelVolatility = Matrix{T}(undef, length(mktData.TimeToMaturity), length(mktData.Strikes))
    for i = 1:length(mktData.TimeToMaturity)
        smile = FinancialMonteCarlo.EuropeanOptionSmile(mktData.TimeToMaturity[i], mktData.Strikes, isCall)
        OutputPriceVec = vec(pricer(calibrated_model, ZeroRate(mktData.ZeroRates[i]), pricing_method, smile, mode))
        ModelVolatility[i, 1:end] = Array(blsimpv.(model.underlying.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, implied_dividend(model.underlying, mktData.TimeToMaturity[i])))
    end
    return CalibratedData(OptimalParameters, result.minimum, ModelVolatility)
end
