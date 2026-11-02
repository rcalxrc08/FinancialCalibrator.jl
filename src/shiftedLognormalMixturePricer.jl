"""
Pricing European Options with Shifted Lognormal Mixture Model

		Price=shiftedLognormalMixture(S0::Number,K::Number,r::Real,T::Real,eta1::AbstractVector{<:Real},lambda1::AbstractVector{<:Real},alpha1::Real,d::Real=0.0,flag::Bool=true)

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
function shiftedLognormalMixture(S0::Number, K::Number, r::Real, T::Real, eta1::Union{AbstractVector{<:Real}, Tuple}, lambda1::Union{AbstractVector{<:Real}, Tuple}, alpha1::Real, d::Real = 0.0, flag::Bool = true)

    ########################################
    Kappa = K - S0*alpha1*exp((r - d)*T)
    A0 = S0*(1 - alpha1)

    if (Kappa <= 0||A0 <= 0||minimum(lambda1) <= 0||sum(lambda1) > 1||minimum(eta1) <= 0)
        price = oftype(float(S0), -100000)
    else
        price=zero(float(S0))
        for j = 1:length(lambda1)
            price+=lambda1[j] * blsprice(A0, Kappa, r, T, eta1[j], d)
        end
        price+=(1-sum(lambda1))*blsprice(A0, Kappa, r, T, eta1[end], d)
        if !flag
            price -= S0*exp(-d*T) - K*exp(-r*T)
        end
    end

    return price
end

"""
    pricer(model::ShiftedLogNormalMixture, zero_rate, ::AnalyticMethod, opt, mode = SerialMode())

Closed-form price of a `EuropeanOption` or of a `EuropeanOptionSmile` under the shifted lognormal mixture. The spot,
the dividend curve and the rate curve are those of the `model` and of `zero_rate`, as for the numerical methods;
the rate and dividend are flat-equivalent over the maturity of `opt`. The strikes of a smile can live on the GPU
(`CuArray`), in which case the prices are computed there in a single broadcast.
"""
function pricer(model::ShiftedLogNormalMixture, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, ::AnalyticMethod, opt::Union{FinancialMonteCarlo.EuropeanOption, FinancialMonteCarlo.EuropeanOptionSmile}, ::FinancialMonteCarlo.BaseMode = FinancialMonteCarlo.SerialMode())
    T = opt.T
    S0 = model.underlying.S0
    # the curves may integrate in a wider type than the model (e.g. Float64 for a Float32 model)
    FT = float(promote_type(typeof(S0), typeof(T)))
    r = FT(FinancialMonteCarlo.integral(zero_rate.r, T) / T)
    d = FT(FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(model), T) / T)
    # tuples can be captured by a GPU broadcast
    eta = Tuple(model.η)
    lambda = Tuple(model.λ)
    return shiftedLognormalMixture.(S0, opt.K, r, T, Ref(eta), Ref(lambda), model.α, d, opt.isCall)
end
