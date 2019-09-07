using Dierckx, Dates,FFTW,Random,FinancialFFT,FinancialMonteCarlo

include("RetrieverCalibration.jl")
include("RetrieverCalibrationMDS.jl")
include("AdapterCalibration.jl")
include("AdapterCalibrationMDS.jl")
include("shiftedLognormalMixturePricer.jl")

import FinancialMonteCarlo.pricer

abstract type CalibratorBase end

struct CalibratorCarrMadan<:CalibratorBase
	Model::FinancialMonteCarlo.BaseProcess
end

function pricer(cal::CalibratorBase,S0::Number,StrikeVec::Array{Float64},r::Float64,T::Float64,Param::Array{Float64},d::Float64=0.0)::Array{Float64}
	error("Not implemented")
end

function pricer(cal::CalibratorCarrMadan,S0::Number,StrikeVec::Array{Float64},r::Float64,T::Float64,Param::Array{Float64},d::Float64=0.0)::Array{Float64}
	EUData=[EuropeanOption(T,K1) for K1 in StrikeVec];
	
	return pricer(cal.Model,equitySpotData(S0,r,d),CarrMadanMethod(400.0,14),EUData);
end

struct CalibratorShiftedLogNormalMixture <:CalibratorBase
	CalibratorShiftedLogNormalMixture()=new(0)
end

function pricer(cal::CalibratorShiftedLogNormalMixture,S0::Number,StrikeVec::Array{Float64},r::Float64,T::Float64,Param::Array{Float64},d::Float64=0.0,AddInput::Integer=0)::Array{Float64}
	return shiftedLognormalMixturePricer(S0,StrikeVec,r,T,Param,d);
end


function Loss(p::Array{Float64},cal::CalibratorBase,mktData::MarketData)::Float64
	loss=0.0;
	try
	@simd for i=1:length(mktData.TimeToMaturity)
		OutputPriceVec=pricer(cal,mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],p,mktData.ImpliedDividend[i]);
		if (minimum(OutputPriceVec)<=0.0)
			return 1000000;
		end
		VolaTmp=blsimpv.(mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],OutputPriceVec,mktData.ImpliedDividend[i]);
		loss+=norm(VolaTmp-mktData.Volatility[i,1:end])
	end
	catch
		loss=1e+4;
	end
	return loss;
end

struct CalibratedData
	OptimalParam::Array{Float64}
	OptimalValue::Float64
	ModelVolatility::Matrix{Float64}
	CalibratedData(OptimalParameters,minima,ModelVolatility)=new(OptimalParameters,minima,ModelVolatility)
end


using FinancialToolbox,Optim;

function calibrate(cal::CalibratorBase,mktData::MarketData,method::Optim.AbstractOptimizer=NelderMead())::CalibratedData
	#const LossFunction(v::Array{Float64})=Loss(v,cal,mktData)
	LossFunction(v::Array{Float64})=Loss(v,cal,mktData)
	model=cal.Model;
	N2=fieldcount(typeof(model));
	fields_=fieldnames(NormalInverseGaussianProcess)
	InitialPoint=zeros(Float64,N2)
	for i=1:N2
		InitialPoint[i]=getfield(model,fields_[i])
	end
	
	result = optimize(LossFunction, InitialPoint,method)
	OptimalParameters=result.minimizer
	ModelVolatility=Matrix{Float64}(undef,length(mktData.TimeToMaturity),length(mktData.Strikes));
	for i=1:length(mktData.TimeToMaturity)
		OutputPriceVec=pricer(cal,mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],OptimalParameters,mktData.ImpliedDividend[i]);
		ModelVolatility[i,1:end]=blsimpv.(mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],OutputPriceVec,mktData.ImpliedDividend[i]);
	end
	return CalibratedData(OptimalParameters,result.minimum,ModelVolatility);
end
