import FinancialMonteCarlo.pricer

# Call prices of the strikes `StrikeVec` at maturity `T` for `model` re-parametrized with `Param`.
function smile_prices(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, Param::AbstractVector{<:Real}, S0::Real, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real, d::Real = 0.0)
    # set_parameters! returns a new model, it does not mutate `model`
    return price_smile(FinancialMonteCarlo.set_parameters!(model, Param), method, S0, StrikeVec, r, T, d)
end

# Numerical: generic over any (model, method) pair supported by FinancialMonteCarlo/FinancialFFT.
function price_smile(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, S0::Real, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real, d::Real)
    smile = FinancialMonteCarlo.EuropeanOptionSmile(T, StrikeVec, fill(true, length(StrikeVec)))
    return pricer(model, ZeroRate(r), method, smile)
end

# Analytical: closed forms, one method per model.
function price_smile(model::ShiftedLogNormalMixture, ::ClosedFormMethod, S0::Real, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real, d::Real)
    return shiftedLognormalMixturePricer(S0, StrikeVec, r, T, FinancialMonteCarlo.get_parameters(model), d)
end

function price_smile(model::FinancialMonteCarlo.BaseProcess, ::ClosedFormMethod, S0::Real, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real, d::Real)
    error("No closed form available for $(typeof(model).name.name)")
end
