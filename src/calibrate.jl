"""
    calibrate(model, pricing_method, mktData::MarketData, opt_method = NelderMead(), opt_options = Optim.Options(); verbose = false)

Calibrate the parameters of `model` to the implied volatility surface in `mktData`. Each smile is priced with
`pricing_method`, any `FinancialMonteCarlo.AbstractMethod`: a FinancialFFT method such as `CarrMadanMethod`,
`AnalyticMethod()` for models with a closed form, a `MonteCarloConfiguration` or a VibratoMonteCarlo method. Its execution
mode moves to `CudaMode` when the strikes of `mktData` are on the GPU. The loss is minimized with the Optim optimizer `opt_method` and settings `opt_options` (iterations, time limit,
tolerances, ...), starting from the current parameters of `model`. Nothing is printed unless `verbose = true`.
"""
function calibrate(model::FinancialMonteCarlo.BaseProcess, pricing_method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, opt_method::Optim.AbstractOptimizer = NelderMead(), opt_options::Optim.Options = Optim.Options(); verbose::Bool = false)::CalibratedData{T} where {T}
    # what does not depend on the parameters is built once, not at every evaluation of the loss
    cache = loss_cache(model, pricing_method, mktData)
    loss_function(v::AbstractVector{<:Real}) = loss(v, model, pricing_method, mktData, cache)
    InitialPoint = FinancialMonteCarlo.get_parameters(model)
    result = optimize(loss_function, InitialPoint, opt_method, opt_options)
    OptimalParameters = result.minimizer
    if verbose
        @show loss_function(OptimalParameters)
        println(result)
    end
    calibrated_model = FinancialMonteCarlo.set_parameters!(model, OptimalParameters)
    ModelVolatility = Matrix{T}(undef, length(mktData.TimeToMaturity), length(mktData.Strikes))
    for i = 1:length(mktData.TimeToMaturity)
        OutputPriceVec = vec(pricer(calibrated_model, cache.zero_rates[i], cache.method, cache.smiles[i]))
        ModelVolatility[i, :] = Array(blsimpv.(model.underlying.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, cache.dividends[i]))
    end
    return CalibratedData(OptimalParameters, result.minimum, ModelVolatility)
end
