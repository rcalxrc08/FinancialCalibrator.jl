
function calibrate(cal::CalibratorBase, mktData::MarketData, method::Optim.AbstractOptimizer = NelderMead())::CalibratedData
    #const LossFunction(v::Array{Float64})=Loss(v,cal,mktData)
    LossFunction(v::Array{Float64}) = Loss(v, cal, mktData)
    model=cal.Model
    model_type=typeof(model)
    N2=fieldcount(model_type)
    fields_=fieldnames(model_type)
    InitialPoint = FinancialMonteCarlo.get_parameters(model)
    result = optimize(LossFunction, InitialPoint, method)
    OptimalParameters=result.minimizer
    @show LossFunction(OptimalParameters)
    println("RESULT")
    println(result)
    println(summary(result))
    ModelVolatility=Matrix{Float64}(undef, length(mktData.TimeToMaturity), length(mktData.Strikes))
    for i = 1:length(mktData.TimeToMaturity)
        OutputPriceVec=pricer(cal, mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OptimalParameters, mktData.ImpliedDividend[i])
        ModelVolatility[i, 1:end]=blsimpv.(mktData.S0, mktData.Strikes, mktData.ZeroRates[i], mktData.TimeToMaturity[i], OutputPriceVec, mktData.ImpliedDividend[i])
    end
    return CalibratedData(OptimalParameters, result.minimum, ModelVolatility)
end
