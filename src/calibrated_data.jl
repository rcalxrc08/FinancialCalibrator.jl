"""
    AnalyticMethod()

Pricing method tag selecting the closed-form pricer of a model, through the `pricer` interface of FinancialMonteCarlo:
`pricer(model, zero_rate, AnalyticMethod(), smile)`. Currently implemented for `ShiftedLogNormalMixture`. Numerical
pricing methods (`CarrMadanMethod`, `CosMethod`, ...) come from FinancialFFT, so the kind of calibration is entirely
determined by the pricing method.
"""
struct AnalyticMethod <: FinancialMonteCarlo.AbstractMethod end

struct CalibratedData{T <: AbstractFloat}
    OptimalParam::Vector{T}
    OptimalValue::T
    ModelVolatility::Matrix{T}
end
CalibratedData(OptimalParameters::AbstractVector{T}, minima::Real, ModelVolatility::AbstractMatrix) where {T <: AbstractFloat} = CalibratedData{T}(OptimalParameters, T(minima), ModelVolatility)
