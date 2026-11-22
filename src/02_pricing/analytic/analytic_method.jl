"""
    AnalyticMethod(; mode = SerialMode())

Pricing method selecting the closed-form pricer of a model, through the `pricer` interface of FinancialMonteCarlo:
`pricer(model, zero_rate, AnalyticMethod(), smile)`. Implemented for `BlackScholesProcess`, `LogNormalMixture` and
their shifted versions, `ShiftedProcess(model, α)` (the displaced diffusion and the shifted lognormal mixture).
Numerical pricing methods (`CarrMadanMethod`, `CosMethod`, ...) come from FinancialFFT, so
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

# The prices have the type computed by `pricer`. The number of components of a mixture is in the type of the model
# (FinancialMonteCarlo's LogNormalMixture holds tuples), so the kernel and its call are inferred: no assertion (#35).
analytic_prices(::Type{PT}, kernel, opt::FinancialMonteCarlo.EuropeanOption) where {PT} = convert(PT, kernel(opt.K, opt.isCall))
# One broadcast that allocates its result, no `.=` into `similar`: Zygote cannot differentiate a mutation ("Mutating
# arrays is not supported"). The conversion to PT is an isbits callable, which a GPU broadcast captures (a closure over
# the type PT is not isbits); Zygote differentiates its fields on the host and, by dual numbers, on the device, where the
# price is a dual: hence the conversion to the promotion of PT and its type, PT for a plain number (#41).
struct PriceAs{PT, K}
    kernel::K
end
@inline function (p::PriceAs{PT})(K, isCall::Bool) where {PT}
    price = p.kernel(K, isCall)
    return convert(promote_type(PT, typeof(price)), price)
end
analytic_prices(::Type{PT}, kernel, opt::FinancialMonteCarlo.EuropeanOptionSmile) where {PT} = PriceAs{PT, typeof(kernel)}(kernel).(opt.K, opt.isCall)

# Every field is typed by a type parameter of its own (the tuples too, not NTuple{N, V}): on the device Zygote
# differentiates the fields of a broadcast callable by rebuilding it with dual numbers, which a field of a composite
# type such as NTuple{N, V} cannot hold (it returned `nothing`, #41), nor a parameter bounded by another one: N is set by
# the outer constructor.
"""
    LognormalMixtureKernel(F, shift, σ, w, r, T)

The price of a European option on ``S_T = A_T + s`` (`s` the shift) where ``A_T`` is a mixture of lognormals of forward
`F`: with weight `w[j]` its volatility is `σ[j]`, each priced by Black's formula (`blkprice` of FinancialToolbox). The
strike of ``A`` is ``κ = K - s``; when ``κ ≤ 0`` the call is a forward (always exercised) and the put is worthless.
"""
struct LognormalMixtureKernel{N, F_, S, Σ, W, R, T_}
    F::F_
    shift::S
    σ::Σ
    w::W
    r::R
    T::T_
end
LognormalMixtureKernel(F, shift, σ::NTuple{N, Any}, w::NTuple{N, Any}, r, T) where {N} = LognormalMixtureKernel{N, typeof(F), typeof(shift), typeof(σ), typeof(w), typeof(r), typeof(T)}(F, shift, σ, w, r, T)

# Σⱼ w[j] blkprice(σ[j]), unrolled at compile time into nested muladds. Not `sum(map(closure, w, σ))`: here Zygote did not
# infer its pullback (a minimal `map` over tuples it does infer, the cause is not pinned down), and one uninferred
# pullback makes the whole gradient dynamic (#39: with the
# `promote` below, the Zygote gradient of a price is 1.4 times the price instead of ~30 times; Mooncake and Enzyme unchanged).
@generated function mixture_price(F, κ, r, T, σ::NTuple{N, Any}, w::NTuple{N, Any}, isCall::Bool) where {N}
    ex = :(w[$N] * blkprice(F, κ, r, T, σ[$N], isCall))
    for j = (N-1):-1:1
        ex = :(muladd(w[$j], blkprice(F, κ, r, T, σ[$j], isCall), $ex))
    end
    return ex
end

@inline function (k::LognormalMixtureKernel)(K, isCall::Bool)
    κ = K - k.shift
    positive = κ > 0
    # any positive strike where κ ≤ 0: that branch is discarded, it must only stay finite (and so its derivatives)
    κ⁺ = ifelse(positive, κ, one(κ))
    mixture = mixture_price(k.F, κ⁺, k.r, k.T, k.σ, k.w, isCall)
    # no branch on isCall (#39): `Bool * x` is a select with a strong zero, straight-line code for every AD backend
    linear = exp(-k.r * k.T) * (isCall * (k.F - κ))
    # `promote`, not `oftype(mixture, linear)`: Zygote's pullback of `oftype` is not inferred, Base's `oftype` has an `isa`
    # branch (FluxML/Zygote.jl#1751, Zygote 0.7.18; #39)
    return ifelse(positive, promote(mixture, linear)...)
end

# The same prices with a stateless callable: the numbers of the kernel are broadcast arguments, not fields. Mooncake's
# CUDA broadcast refuses a callable with differentiable fields (#41 group 3c), so its rule for `analytic_prices` on the
# device (FinancialCalibratorCUDAMooncakeExt) differentiates this function instead. Not the default: Zygote's device
# broadcast infers its dual numbers for at most 10 arguments, a mixture of N components has 2N + 6.
struct MixturePrice{PT, N} end
@generated function (::MixturePrice{PT, N})(K, isCall::Bool, F, shift, r, T, σw::Vararg{Any, M}) where {PT, N, M}
    σ = Expr(:tuple, (:(σw[$j]) for j = 1:N)...)
    w = Expr(:tuple, (:(σw[$(N+j)]) for j = 1:N)...)
    return quote
        kernel = LognormalMixtureKernel(F, shift, $σ, $w, r, T)
        return PriceAs{PT, typeof(kernel)}(kernel)(K, isCall)
    end
end
stateless_prices(::Type{PT}, k::LognormalMixtureKernel{N}, opt::FinancialMonteCarlo.EuropeanOptionSmile) where {PT, N} = MixturePrice{PT, N}().(opt.K, opt.isCall, k.F, k.shift, k.r, k.T, k.σ..., k.w...)
