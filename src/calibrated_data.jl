"""
    ClosedFormMethod()

Pricing method tag selecting a closed-form smile price for models that have one
(e.g. `ShiftedLogNormalMixture`). Numerical pricing methods (`CarrMadanMethod`, `CosMethod`, ...) come
from FinancialFFT, so the kind of calibration is entirely determined by the pricing method.
"""
struct ClosedFormMethod <: FinancialMonteCarlo.AbstractMethod end

struct CalibratedData
    OptimalParam::Array{Float64}
    OptimalValue::Float64
    ModelVolatility::Matrix{Float64}
    CalibratedData(OptimalParameters, minima, ModelVolatility) = new(OptimalParameters, minima, ModelVolatility)
end
