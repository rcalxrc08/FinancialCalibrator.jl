# Mooncake on the device, closed form (#41 group 3c): Mooncake's CUDA broadcast refuses a callable with differentiable
# fields, and the closed form broadcasts the kernel (`PriceAs`) over the strikes. Its rule differentiates the same prices
# computed by a stateless callable, the numbers of the kernel passed as broadcast arguments (`stateless_prices`). Only
# on the device and only for Mooncake: the callable stays the default, since Zygote's device broadcast needs it and a
# stateless broadcast made host Zygote 3.4x slower. Remove if Mooncake differentiates the fields of a broadcast callable.
module FinancialCalibratorCUDAMooncakeExt

using CUDA: CuArray
using FinancialCalibrator: FinancialCalibrator, LognormalMixtureKernel, stateless_prices
using FinancialMonteCarlo: EuropeanOptionSmile
using Mooncake: Mooncake, CoDual, NoRData, primal, build_rrule, zero_fcodual, @is_primitive, DefaultCtx, ReverseMode

@is_primitive DefaultCtx ReverseMode Tuple{typeof(FinancialCalibrator.analytic_prices), Type, LognormalMixtureKernel, EuropeanOptionSmile{<:Any, <:Any, <:CuArray}}

function Mooncake.rrule!!(::CoDual{typeof(FinancialCalibrator.analytic_prices)}, PT::CoDual, kernel::CoDual, opt::CoDual)
    # built once per signature: build_rrule caches it and returns a copy
    rule = build_rrule(stateless_prices, primal(PT), primal(kernel), primal(opt))
    prices, prices_pullback = rule(zero_fcodual(stateless_prices), PT, kernel, opt)
    analytic_prices_pullback(dy) = (NoRData(), Base.tail(prices_pullback(dy))...)
    return prices, analytic_prices_pullback
end

end # module
