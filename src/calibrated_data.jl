"""
    ClosedFormMethod()

Pricing method tag selecting a closed-form smile price for models that have one
(e.g. `ShiftedLogNormalMixture`). Numerical pricing methods (`CarrMadanMethod`, `CosMethod`, ...) come
from FinancialFFT, so the kind of calibration is entirely determined by the pricing method.
"""
struct ClosedFormMethod <: FinancialMonteCarlo.AbstractMethod end

struct CalibratedData{T <: AbstractFloat}
    OptimalParam::Vector{T}
    OptimalValue::T
    ModelVolatility::Matrix{T}
end
CalibratedData(OptimalParameters::AbstractVector{T}, minima::Real, ModelVolatility::AbstractMatrix) where {T <: AbstractFloat} = CalibratedData{T}(OptimalParameters, T(minima), ModelVolatility)
