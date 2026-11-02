
abstract type CalibratorBase end

struct CalibratorCarrMadan{T0 <: FinancialMonteCarlo.BaseProcess, M <: CarrMadanMethod} <: CalibratorBase
    Model::T0
    method_::M
end
CalibratorCarrMadan(model::FinancialMonteCarlo.BaseProcess, A::Real = 400.0, Npow::Integer = 14, alpha::Real = 0.75) = CalibratorCarrMadan(model, CarrMadanMethod(A, Npow, alpha))

struct CalibratorShiftedLogNormalMixture{T0 <: ShiftedLogNormalMixture} <: CalibratorBase
    Model::T0
end
CalibratorShiftedLogNormalMixture(eta, lambda, alpha, underlying::FinancialMonteCarlo.AbstractUnderlying) = CalibratorShiftedLogNormalMixture(ShiftedLogNormalMixture(eta, lambda, alpha, underlying))

struct CalibratedData
    OptimalParam::Array{Float64}
    OptimalValue::Float64
    ModelVolatility::Matrix{Float64}
    CalibratedData(OptimalParameters, minima, ModelVolatility) = new(OptimalParameters, minima, ModelVolatility)
end
