using Dierckx, Dates,FFTW,Random,FinancialFFT,FinancialMonteCarlo, LinearAlgebra

function Loss(params::Array{Float64},cal::CalibratorBase,mktData::MarketData)::Float64
	loss=0.0;
	try
	@simd for i=1:length(mktData.TimeToMaturity)
		OutputPriceVec=pricer(cal,mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],params,mktData.ImpliedDividend[i]);
		if (minimum(OutputPriceVec)<=0.0)
			@show "negative price"
			return 1000000;
		end
		VolaTmp=blsimpv.(mktData.S0,mktData.Strikes,mktData.ZeroRates[i],mktData.TimeToMaturity[i],OutputPriceVec,mktData.ImpliedDividend[i]);
		loss+=norm(VolaTmp-mktData.Volatility[i,1:end])
	end
	catch e
		@show e
		loss=1e+4;
	end
	return loss;
end