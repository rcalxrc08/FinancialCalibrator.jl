
# The spot and the dividends are those of the underlying of `model`, the rate comes from `r`.
# Call prices of the strikes `StrikeVec` at maturity `T` for `model` re-parametrized with `Param`.
function smile_prices(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, Param::AbstractVector{<:Real}, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real)
    # set_parameters! returns a new model, it does not mutate `model`
    return price_smile(FinancialMonteCarlo.set_parameters!(model, Param), method, StrikeVec, r, T)
end

# Execution mode used to price strikes stored in `StrikeVec`: serial on the host; the CUDA extension
# selects `CudaMode` for a `CuArray`.
pricing_mode(::AbstractArray) = FinancialMonteCarlo.SerialMode()

# Numerical: generic over any (model, method) pair supported by FinancialMonteCarlo/FinancialFFT.
price_smile(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real) = numerical_price_smile(model, method, StrikeVec, r, T)

function numerical_price_smile(model, method, StrikeVec, r, T)
    # the call flags live where the strikes live
    isCall = fill!(similar(StrikeVec, Bool), true)
    smile = FinancialMonteCarlo.EuropeanOptionSmile(T, StrikeVec, isCall)
    return vec(pricer(model, ZeroRate(r), method, smile, pricing_mode(StrikeVec)))
end

# Analytical: closed forms, one per model, through the same `pricer` interface.
price_smile(model::ShiftedLogNormalMixture, method::AnalyticMethod, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real) = numerical_price_smile(model, method, StrikeVec, r, T)

function price_smile(model::FinancialMonteCarlo.BaseProcess, ::AnalyticMethod, StrikeVec::AbstractVector{<:Real}, r::Real, T::Real)
    error("No closed form available for $(typeof(model).name.name)")
end
