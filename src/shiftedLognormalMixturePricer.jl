import Dierckx.Spline1D;
"""
Pricing European Options with Shifted Lognormal Mixture Model

		Price=shiftedLognormalMixture(S0::Number,K::Number,r::Real,T::Real,eta1::Array{Float64},lambda1::Array{Float64},alpha1::Real,d::Real=0.0,flag::Bool=true)::Float64

Where:\n
		S0 = Spot Price.
		K = Strike Price.
		r= zero rate with tenor T.
		T= tenor of the options.
		eta1= Vector of Volatility of "fake" underlying.
		lambda1= Vector of weights for "fake" underlying.
		alpha1= Shifting parameter.
		d= implied dividend.
		flag= true if is call option, false otherwise.

		Price= Price of the European Option with Strike equals to K, tenor T.
"""
function shiftedLognormalMixture(S0::Number,K::Number,r::Real,T::Real,eta1::Array{Float64},lambda1::Array{Float64},alpha1::Real,d::Real=0.0,flag::Bool=true)::Float64

	########################################
	Kappa = K - S0*alpha1*exp((r - d)*T);
	A0 = S0*(1. - alpha1);

	if (Kappa <= 0.0||A0<=0.0||minimum(lambda1)<=0.0||sum(lambda1)>1.0||minimum(eta1)<=0.0)
		price = -100000;
	else
		price=0.0;
		for j = 1: length(lambda1)
			price+=lambda1[j] *blsprice(A0, Kappa,r, T, eta1[j], d );
		end
		price+=(1.0-sum(lambda1))*blsprice(A0, Kappa, r,T, eta1[end], d )
		if !flag
			price -= S0*exp(-d*T) - K*exp(-r*T);
		end
	end

	return price;
end


function shiftedLognormalMixturePricer(S0::Number,StrikeVec::Array{Float64},r::Real,T::Real,Param::Array{Float64},d::Real=0.0,AdditionalInput=0)::Array{Float64}
	
	Neta=div(length(Param),2);
	Nlam=Neta-1;
	eta1=Param[1:Neta];
	lambda1=Param[Neta+1:end-1];
	alpha1=Param[end];

	PriceVec=[shiftedLognormalMixture(S0,strike,r,T,eta1,lambda1,alpha1,d) for strike in StrikeVec];

	return PriceVec;
end
