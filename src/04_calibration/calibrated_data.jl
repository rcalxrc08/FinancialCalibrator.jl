"""
    CalibratedData

The result of [`calibrate`](@ref): `OptimalParam`, the calibrated parameters of the model (in the order of
`FinancialMonteCarlo.get_parameters`), `OptimalValue`, the loss there, and `ModelVolatility`, the implied volatilities
of the calibrated model at the quotes, shaped as the surface (`NaN` at the quotes left out of the loss).
`Converged` is `Optim.converged` of the optimizer (before the polish), `Iterations` and `LossCalls` its iterations and
its evaluations of the loss (`Optim.iterations`, `Optim.f_calls`): `calibrate` warns when the optimizer stopped at its
iteration or time limit.
"""
struct CalibratedData{T <: AbstractFloat}
    OptimalParam::Vector{T}
    OptimalValue::T
    ModelVolatility::Matrix{T}
    Converged::Bool
    Iterations::Int
    LossCalls::Int
end
CalibratedData(OptimalParameters::AbstractVector{T}, minima::Real, ModelVolatility::AbstractMatrix, converged::Bool, iterations::Integer, loss_calls::Integer) where {T <: AbstractFloat} = CalibratedData{T}(OptimalParameters, T(minima), ModelVolatility, converged, iterations, loss_calls)
