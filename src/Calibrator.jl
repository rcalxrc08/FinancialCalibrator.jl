using Dierckx, Dates,FFTW,Random,FinancialFFT,FinancialMonteCarlo

abstract type CalibratorBase end

struct CalibratorCarrMadan{ T0<: FinancialMonteCarlo.LevyProcess, num <: Number, num1 <: Number}<:CalibratorBase
	Model::T0
	method_::CarrMadanMethod{num,num1}
	#function CalibratorCarrMadan(model::t0,A::num_0=400.0,Npow::num_1=14) where {t0 <: FinancialMonteCarlo.LevyProcess, num_0 <: AbstractFloat, num_1 <: Integer}
    #    if A <= 0.0
    #        error("A must be positive")
    #    elseif Npow <= 2
    #        error("Npow must be greater than 2")
    #    else
    #        return new{t0,num_0,num_1}(model,CarrMadanMethod(A,Npow))
    #    end
    #end
	CalibratorCarrMadan(model::t0,A::num_0=400.0,Npow::num_1=14) where {t0 <: FinancialMonteCarlo.LevyProcess, num_0 <: AbstractFloat, num_1 <: Integer} = new{t0,num_0,num_1}(model,CarrMadanMethod(A,Npow))
end

struct CalibratorShiftedLogNormalMixture <:CalibratorBase
	Model::ShiftedLogNormalMixture
	CalibratorShiftedLogNormalMixture(eta,mu,alfa)=new(ShiftedLogNormalMixture(eta,mu,alfa));
end

struct CalibratedData
	OptimalParam::Array{Float64}
	OptimalValue::Float64
	ModelVolatility::Matrix{Float64}
	CalibratedData(OptimalParameters,minima,ModelVolatility)=new(OptimalParameters,minima,ModelVolatility)
end

include("RetrieverCalibration.jl")
include("AdapterCalibration.jl")
include("shiftedLognormalMixturePricer.jl")
include("pricer.jl")
include("loss.jl")

using FinancialToolbox,Optim;

function calibrate(cal::CalibratorBase,mktData::MarketData,method::Optim.AbstractOptimizer=NelderMead())::CalibratedData
	#const LossFunction(v::Array{Float64})=Loss(v,cal,mktData)
	LossFunction(v::Array{Float64})=Loss(v,cal,mktData)
	model=cal.Model;
	model_type=typeof(model);
	N2=fieldcount(model_type);
	fields_=fieldnames(model_type)
	InitialPoint = FinancialMonteCarlo.get_parameters(model);
	result = optimize(LossFunction, InitialPoint,method)
	OptimalParameters=result.minimizer
	@show LossFunction(OptimalParameters)
	println("RESULT")
	println(result)
	println(summary(result))
	ModelVolatility=Matrix{Float64}(undef,length(mktData.TimeToMaturity),length(mktData.Strikes));
	for i=1:length(mktData.TimeToMaturity)
		OutputPriceVec=pricer(cal,mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],OptimalParameters,mktData.ImpliedDividend[i]);
		ModelVolatility[i,1:end]=blsimpv.(mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],OutputPriceVec,mktData.ImpliedDividend[i]);
	end
	return CalibratedData(OptimalParameters,result.minimum,ModelVolatility);
end
