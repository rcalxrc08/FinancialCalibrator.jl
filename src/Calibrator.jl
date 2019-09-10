using Dierckx, Dates,FFTW,Random,FinancialFFT,FinancialMonteCarlo

include("RetrieverCalibration.jl")
include("AdapterCalibration.jl")
include("shiftedLognormalMixturePricer.jl")
include("pricer.jl")
include("loss.jl")

abstract type CalibratorBase end

struct CalibratorCarrMadan<:CalibratorBase
	Model::FinancialMonteCarlo.BaseProcess
end

struct CalibratorShiftedLogNormalMixture <:CalibratorBase
	CalibratorShiftedLogNormalMixture()=new(0)
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
