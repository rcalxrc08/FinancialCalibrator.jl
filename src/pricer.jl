using Dierckx, Dates,FFTW,Random,FinancialFFT,FinancialMonteCarlo

import FinancialMonteCarlo.pricer

function pricer(cal::CalibratorBase,S0::Number,StrikeVec::Array{Float64},r::Float64,T::Float64,Param::Array{Float64},d::Float64=0.0)::Array{Float64}
	error("Not implemented")
end

function pricer(cal::CalibratorCarrMadan,S0::Number,StrikeVec::Array{Float64},r::Float64,T::Float64,Param::Array{Float64},d::Float64=0.0)::Array{Float64}
	EUData=[EuropeanOption(T,K1) for K1 in StrikeVec];
	prevModel=cal.Model;
	FinancialMonteCarlo.set_parameters!(prevModel,Param);
	
	return pricer(prevModel,ZeroRate(r),CarrMadanMethod(400.0,14),EUData);
end

function pricer(cal::CalibratorShiftedLogNormalMixture,S0::Number,StrikeVec::Array{Float64},r::Float64,T::Float64,Param::Array{Float64},d::Float64=0.0,AddInput::Integer=0)::Array{Float64}
	return shiftedLognormalMixturePricer(S0,StrikeVec,r,T,Param,d);
end