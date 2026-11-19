# Everything the loss needs that depends neither on the parameters nor on the optimizer iteration: the quotes that enter
# it, the smiles, the rates and the dividends of each maturity, and the pricing method in the mode of the strikes
# (calibration_method: CudaMode for a CuArray). Built once per calibration: the with_mode of a MonteCarloConfiguration
# rebuilds its random state.
# Each smile holds out of the money options only, puts below the forward and calls above it: an in the money option is
# mostly intrinsic value, and its implied volatility is ill-conditioned. For the same reason the quotes whose vega, at the
# quoted volatility, is below `min_vega * S0` are left out: their prices are so small that the error of a numerical
# method is a large error in volatility, and it moves erratically with the parameters (on the DAX surface one such quote,
# priced 1e-10, keeps the loss from being smooth at the 1e-6 level).
function loss_cache(model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData; min_vega::Real = 1e-3)
    S0 = model.underlying.S0
    maturities, rates, strikes = mktData.TimeToMaturity, mktData.ZeroRates, mktData.Strikes
    dividends = [implied_dividend(model.underlying, t) for t in maturities]
    # the indices of the quotes kept, on the host; the strikes and the call flags live where the strikes live
    quoted = [findall(Array(blsvega.(S0, strikes, rates[i], maturities[i], @view(mktData.Volatility[i, :]), dividends[i]) .>= min_vega * S0)) for i in eachindex(maturities)]
    quoted_strikes = [strikes[q] for q in quoted]
    is_call = [k .>= S0 * exp((r - d) * t) for (k, t, r, d) in zip(quoted_strikes, maturities, rates, dividends)]
    smiles = [FinancialMonteCarlo.EuropeanOptionSmile(t, k, c) for (t, k, c) in zip(maturities, quoted_strikes, is_call)]
    # the maturities that have quotes, and the quotes of all of them in one array each (strikes, rates, maturities, dividends,
    # call flags, and the position of each quote in the matrix of volatilities), so that the implied volatilities of all the
    # maturities are one broadcast and their errors one subtraction, without concatenating them
    # Nothing here mutates an array: Zygote traces loss_cache when it differentiates calibrated_parameters (through
    # implicit_calibration) and rejects any copyto!/fill!, even of constants. The positions stay on the host: indexing
    # the surface with them moves them to the device, without a bounds check there (see maturity_errors).
    active = [i for i in eachindex(quoted) if !isempty(quoted[i])]
    repeated(values) = reduce(vcat, [zero(quoted_strikes[i]) .+ values[i] for i in active])
    positions = reduce(vcat, [vec(LinearIndices(mktData.Volatility)[i, quoted[i]]) for i in active])
    flat = (strikes = reduce(vcat, quoted_strikes[active]), rates = repeated(rates), maturities = repeated(maturities), dividends = repeated(dividends), is_call = reduce(vcat, is_call[active]), positions = positions)
    return (quoted = quoted, strikes = quoted_strikes, smiles = smiles, is_call = is_call, zero_rates = [ZeroRate(r) for r in rates], dividends = dividends, method = calibration_method(method, strikes), active = active, flat = flat)
end

"""
    vol_errors(params, model, method, mktData, cache = loss_cache(model, method, mktData); vols = mktData.Volatility)

Model implied volatilities minus the quoted ones, for the parameters `params` of `model`: one entry per quote that enters
the loss (see `loss_cache`: out of the money options, vega at least `min_vega * S0`), maturity by maturity. Generic in the
number type of `params` and of `vols`, so that it can be differentiated with respect to both. Throws on invalid
parameters, and a `DomainError` on a price that is not positive.
"""
function vol_errors(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData, cache = loss_cache(model, method, mktData); vols::AbstractMatrix{<:Real} = mktData.Volatility)
    # set_parameters! returns a new model, it does not mutate `model` (inferred since FinancialMonteCarlo 0.7; the loop
    # over the maturities stays a function of its own, a barrier should a model's type not be inferred).
    new_model = FinancialMonteCarlo.set_parameters!(model, params)
    # the positions of the quotes in the cache index the surface of mktData: checked here on the sizes, on the host, so that
    # maturity_errors reads them without a bounds check, which on the GPU is a reduction read back to the host (a sync)
    ChainRulesCore.@ignore_derivatives size(vols) == size(mktData.Volatility) || throw(DimensionMismatch("vols is $(size(vols)), the surface of mktData $(size(mktData.Volatility))"))
    return maturity_errors(new_model, model.underlying.S0, cache, vols)
end

# The comprehension over the maturities is inferred with dual numbers (tested with `@inferred`): the pricer of a maturity
# is a function of its own, so its return type is known.
function maturity_errors(new_model, S0, cache, vols)
    # all the maturities are priced first, and the prices checked once: the check reads a reduction of the prices back to
    # the host, a synchronization of the GPU that one per maturity would repeat. Then one broadcast for the implied
    # volatilities of all the quotes
    prices = reduce(vcat, [model_prices(new_model, cache, i) for i in cache.active])
    check_prices(prices)
    flat = cache.flat
    # in bounds: the positions come from LinearIndices of the surface, whose size vol_errors has checked
    return blsimpv.(S0, flat.strikes, flat.rates, flat.maturities, prices, flat.dividends, flat.is_call) .- @inbounds(vols[flat.positions])
end

# The model prices of the quotes of maturity i that enter the loss.
model_prices(new_model, cache, i::Integer) = vec(pricer(new_model, cache.zero_rates[i], cache.method, cache.smiles[i]))

# the checks on the differentiated path are validations, ignored by the AD backends (their branches are not traced, #39)
check_prices(prices) = ChainRulesCore.@ignore_derivatives minimum(prices) > 0 || throw(DomainError(minimum(prices), "a model price is not positive"))

implied_volatilities(S0, mktData::MarketData, cache, prices, i::Integer) = blsimpv.(S0, cache.strikes[i], mktData.ZeroRates[i], mktData.TimeToMaturity[i], prices, cache.dividends[i], cache.is_call[i])

# The model implied volatilities of the quotes of maturity i that enter the loss.
function model_volatilities(new_model, S0, mktData::MarketData, cache, i::Integer)
    prices = model_prices(new_model, cache, i)
    check_prices(prices)
    return implied_volatilities(S0, mktData, cache, prices, i)
end

"""
    loss(params, model, method, mktData, cache = loss_cache(model, method, mktData))

Half the sum of the squared volatility errors of [`vol_errors`](@ref): smooth in the parameters, also where a maturity
is fitted exactly. `Inf` where the parameters cannot be priced (see `numerical_failure`), which an Optim optimizer ranks
as the worst point; any other exception propagates.

ForwardDiff, Zygote and Enzyme differentiate it (Enzyme with `set_runtime_activity` when the underlying holds a
dividend curve, as `adapter` builds it: FinancialMonteCarlo#79). Mooncake does not: it rejects the `try`/`catch` in
reverse mode (#40), so with Mooncake as `autodiff` use a derivative-free optimizer, or differentiate
[`vol_errors`](@ref), which has no `try`.
"""
loss(params::AbstractVector{<:Real}, model::FinancialMonteCarlo.BaseProcess, method::FinancialMonteCarlo.AbstractMethod, mktData::MarketData{T}, cache = loss_cache(model, method, mktData)) where {T} = loss(promote_type(eltype(params), T), params, model, method, mktData, cache)

# the type of the result as a parameter, so that the loss and the Inf of a failure have the same type. No return type
# assertion: vol_errors is inferred (Float64 and dual numbers, host and device), so the conversion is too.
function loss(::Type{R}, params, model, method, mktData, cache) where {R}
    try
        return R(sum(abs2, vol_errors(params, model, method, mktData, cache)) / 2)
    catch e
        numerical_failure(e) || rethrow()
        # enable with JULIA_DEBUG=FinancialCalibrator to see why
        @debug "loss: parameters that cannot be priced, Inf" params exception = e
        return R(Inf)
    end
end

# The exceptions that mean "these parameters cannot be priced", not a bug: a price that is not positive or has no implied
# volatility, or a model constructor rejecting its parameters: both DomainErrors (FinancialMonteCarlo ≥ 0.8 throws a
# DomainError for every parameter out of its domain, #31, FinancialMonteCarlo#71; an AssertionError is now a bug, as
# is the DimensionMismatch of a wrong number of parameters). The default parametrizations keep the constructors from
# failing (the shift of a ShiftedProcess included, α = 1 - e^{-x}); with Parametrization() they can.
# Anything else (a MethodError after an API change, ...) is a bug and propagates (#3). A constant penalty used to stand
# in for the loss there (#24): a plateau that a line search can stall on, and that looked like a value.
numerical_failure(e) = e isa DomainError
