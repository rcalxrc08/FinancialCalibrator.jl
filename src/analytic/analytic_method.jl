"""
    AnalyticMethod(; mode = SerialMode())

Pricing method selecting the closed-form pricer of a model, through the `pricer` interface of FinancialMonteCarlo:
`pricer(model, zero_rate, AnalyticMethod(), smile)`. Implemented for `BlackScholesProcess` and
`ShiftedLogNormalMixture`. Numerical pricing methods (`CarrMadanMethod`, `CosMethod`, ...) come from FinancialFFT, so
the kind of calibration is entirely determined by the pricing method. Like every method of FinancialMonteCarlo it
carries its execution mode: `FinancialMonteCarlo.CudaMode()` prices the strikes of a smile on the GPU, in one broadcast.
"""
struct AnalyticMethod{modeType <: FinancialMonteCarlo.AbstractMode} <: FinancialMonteCarlo.AbstractMethod
    mode::modeType
end
AnalyticMethod(; mode::FinancialMonteCarlo.AbstractMode = FinancialMonteCarlo.SerialMode()) = AnalyticMethod(mode)

# A model with a closed form defines two methods:
#   analytic_price_type(model): the number type of its prices before the market data enters (inferred: it is used to
#       allocate the prices, so that `pricer` is inferred even when the kernel is not);
#   analytic_kernel(model, S0, r, d, T): a callable (K, isCall) -> price, an isbits struct so that it is captured by a
#       GPU broadcast, with r and d the flat rates equivalent to the curves over the maturity T.
# Everything that depends on the parameters and not on the strike is computed once, in the kernel.

"""
    pricer(model, zero_rate, method::AnalyticMethod, opt)

Closed-form price of a `EuropeanOption` or of a `EuropeanOptionSmile`. The spot, the dividend curve and the rate curve
are those of the `model` and of `zero_rate`, as for the numerical methods; the rate and the dividend are flat-equivalent
over the maturity of `opt`. The strikes of a smile go to the device of the mode of `method`: in `CudaMode` the prices are
computed on the GPU, one thread per strike.
"""
function pricer(model::FinancialMonteCarlo.BaseProcess, zero_rate::FinancialMonteCarlo.AbstractZeroRateCurve, method::AnalyticMethod, opt_::Union{FinancialMonteCarlo.EuropeanOption, FinancialMonteCarlo.EuropeanOptionSmile})
    opt = FinancialMonteCarlo.to_mode(opt_, FinancialMonteCarlo.execution_mode(method))
    T = opt.T
    S0 = model.underlying.S0
    # the curves may integrate in a wider type than the model (e.g. Float64 for a Float32 model)
    FT = float(promote_type(typeof(S0), typeof(T)))
    r = FT(FinancialMonteCarlo.integral(zero_rate.r, T) / T)
    d = FT(FinancialMonteCarlo.integral(FinancialMonteCarlo.dividend(model), T) / T)
    PT = float(promote_type(analytic_price_type(model), FT, eltype(opt.K)))
    return analytic_prices(PT, analytic_kernel(model, S0, r, d, T), opt)
end

# The kernel's type may depend on run time values (the number of components of a mixture): these functions are the
# barrier, and they return the price type computed by `pricer`.
analytic_prices(::Type{PT}, kernel, opt::FinancialMonteCarlo.EuropeanOption) where {PT} = convert(PT, kernel(opt.K, opt.isCall))::PT
function analytic_prices(::Type{PT}, kernel, opt::FinancialMonteCarlo.EuropeanOptionSmile) where {PT}
    prices = similar(opt.K, PT)
    prices .= kernel.(opt.K, opt.isCall)
    return prices
end

"""
    LognormalMixtureKernel(F, shift, σ, w, r, T)

The price of a European option on ``S_T = A_T + s`` (`s` the shift) where ``A_T`` is a mixture of lognormals of forward
`F`: with weight `w[j]` its volatility is `σ[j]`, each priced by Black's formula (`blkprice` of FinancialToolbox). The
strike of ``A`` is ``κ = K - s``; when ``κ ≤ 0`` the call is a forward (always exercised) and the put is worthless.
"""
struct LognormalMixtureKernel{N, F_, S, V, W, R, T_}
    F::F_
    shift::S
    σ::NTuple{N, V}
    w::NTuple{N, W}
    r::R
    T::T_
end

@inline function (k::LognormalMixtureKernel)(K, isCall::Bool)
    κ = K - k.shift
    positive = κ > 0
    # any positive strike where κ ≤ 0: that branch is discarded, it must only stay finite (and so its derivatives)
    κ⁺ = ifelse(positive, κ, one(κ))
    mixture = sum(map((w, σ) -> w * blkprice(k.F, κ⁺, k.r, k.T, σ, isCall), k.w, k.σ))
    linear = exp(-k.r * k.T) * (isCall ? k.F - κ : zero(k.F - κ))
    return ifelse(positive, mixture, oftype(mixture, linear))
end
