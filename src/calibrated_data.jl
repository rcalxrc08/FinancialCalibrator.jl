"""
    AnalyticMethod(; mode = SerialMode())

Pricing method selecting the closed-form pricer of a model, through the `pricer` interface of FinancialMonteCarlo:
`pricer(model, zero_rate, AnalyticMethod(), smile)`. Currently implemented for `ShiftedLogNormalMixture`. Numerical
pricing methods (`CarrMadanMethod`, `CosMethod`, ...) come from FinancialFFT, so the kind of calibration is entirely
determined by the pricing method. Like every method of FinancialMonteCarlo 0.6 it carries its execution mode:
`FinancialMonteCarlo.CudaMode()` prices the strikes of a smile on the GPU.
"""
struct AnalyticMethod{modeType <: FinancialMonteCarlo.AbstractMode} <: FinancialMonteCarlo.AbstractMethod
    mode::modeType
end
AnalyticMethod(; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = AnalyticMethod(mode)

"""
    CalibratedData

The result of [`calibrate`](@ref): `OptimalParam`, the calibrated parameters of the model (in the order of
`FinancialMonteCarlo.get_parameters`), `OptimalValue`, the loss there, and `ModelVolatility`, the implied volatilities
of the calibrated model at the quotes, shaped as the surface (`NaN` at the quotes left out of the loss).
"""
struct CalibratedData{T <: AbstractFloat}
    OptimalParam::Vector{T}
    OptimalValue::T
    ModelVolatility::Matrix{T}
end
CalibratedData(OptimalParameters::AbstractVector{T}, minima::Real, ModelVolatility::AbstractMatrix) where {T <: AbstractFloat} = CalibratedData{T}(OptimalParameters, T(minima), ModelVolatility)
