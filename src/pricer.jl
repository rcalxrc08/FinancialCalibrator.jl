
import FinancialMonteCarlo.pricer

function pricer(cal::CalibratorBase, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, Param::Array{Float64}, d::Float64=0.0)::Array{Float64}
    error("Not implemented")
end

function pricer(cal::CalibratorCarrMadan, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, Param::Array{Float64}, d::Float64=0.0)::Array{Float64}
    smile = FinancialMonteCarlo.EuropeanOptionSmile(T, StrikeVec, fill(true, length(StrikeVec)))
    # set_parameters! returns a new model, it does not mutate cal.Model
    model = FinancialMonteCarlo.set_parameters!(cal.Model, Param)
    return pricer(model, ZeroRate(r), cal.method_, smile)
end

function pricer(cal::CalibratorShiftedLogNormalMixture, S0::Number, StrikeVec::Array{Float64}, r::Float64, T::Float64, Param::Array{Float64}, d::Float64=0.0, AddInput::Integer=0)::Array{Float64}
    return shiftedLognormalMixturePricer(S0, StrikeVec, r, T, Param, d)
end
