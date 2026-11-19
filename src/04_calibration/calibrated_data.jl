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
