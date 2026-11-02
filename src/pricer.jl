import FinancialMonteCarlo.pricer

# Call prices of the strikes `StrikeVec` at maturity `T` for `model` re-parametrized with `Param`.
function smile_prices(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, Param::Array{Float64}, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, d::Float64 = 0.0)::Array{Float64}
    # set_parameters! returns a new model, it does not mutate `model`
    return price_smile(FinancialMonteCarlo.set_parameters!(model, Param), method, S0, StrikeVec, r, T, d)
end

# Numerical: generic over any (model, method) pair supported by FinancialMonteCarlo/FinancialFFT.
function price_smile(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, d::Float64)
    smile = FinancialMonteCarlo.EuropeanOptionSmile(T, StrikeVec, fill(true, length(StrikeVec)))
    return pricer(model, ZeroRate(r), method, smile)
end

# Analytical: closed forms, one method per model.
function price_smile(model::ShiftedLogNormalMixture, ::ClosedFormMethod, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, d::Float64)
    return shiftedLognormalMixturePricer(S0, StrikeVec, r, T, FinancialMonteCarlo.get_parameters(model), d)
end

function price_smile(model::FinancialMonteCarlo.BaseProcess, ::ClosedFormMethod, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, d::Float64)
    error("No closed form available for $(typeof(model).name.name)")
end
