# Market data from the delayed quotes of the CBOE (https://www.cboe.com/delayed_quotes/), free and without a key: the
# option chain of an underlying with the implied volatility of every contract, and the spot. US underlyings only, the
# current snapshot only (15 minutes delayed). Two input strategies of `retriever`: `CboeWebInputStrategy` downloads the
# current chain (and can save it), `CboeInputStrategy` reads a saved one, so that a calibration is reproducible and the
# tests stay offline. `cboe_snapshot` only saves.

"""
    CboeInputStrategy(; rate = 0.04, moneyness = 0.8:0.05:1.2, root = nothing, dir = DATA_DIR)

Input strategy of [`retriever`](@ref) for an option chain of the CBOE delayed quotes, saved by [`cboe_snapshot`](@ref)
as `CboeOptions_<underlying><ddMonyyyy>.json` in `dir`. The chain becomes the inputs of [`adapter`](@ref):

- the spot, from the chain;
- a flat continuously compounded zero rate `rate`, for every expiry (the CBOE gives no curve);
- the forward and the dividend yield of every expiry, implied by put–call parity: ``F = K + e^{rT} (C - P)`` from the
  mid prices of the strikes nearest to the spot, and ``q = r - \\log(F / S_0) / T``;
- the implied volatilities of the CBOE (`iv`) of the out of the money options (puts below the forward, calls above),
  with a positive bid and an implied volatility between 0.1% and 500%, interpolated linearly in moneyness ``K / S_0`` on
  the grid `moneyness`. The expiries whose quotes do not cover the grid are left out, so that every maturity has the
  same strikes.

`root` selects the option root of the chain (`"SPX"`, the monthly options of the S&P 500, or `"SPXW"`, the weeklies);
by default it is the underlying of the `CalibrationID`.
"""
struct CboeInputStrategy{R <: Real, M <: AbstractVector{<:Real}} <: InputType
    rate::R
    moneyness::M
    root::Union{Nothing, String}
    dir::String
end
CboeInputStrategy(; rate::Real = 0.04, moneyness::AbstractVector{<:Real} = 0.8:0.05:1.2, root::Union{Nothing, String} = nothing, dir::AbstractString = DATA_DIR) = CboeInputStrategy(rate, moneyness, root, String(dir))

"""
    CboeWebInputStrategy(; rate = 0.04, moneyness = 0.8:0.05:1.2, root = nothing, index = true, dir = nothing)

Input strategy of [`retriever`](@ref) that downloads the current option chain of the underlying of the
`CalibrationID` from the CBOE delayed quotes (an index, `index = true`, such as `"SPX"`, is requested as `_SPX`), and
turns it into the inputs of [`adapter`](@ref) as [`CboeInputStrategy`](@ref) does, with the same `rate`, `moneyness`
and `root`. The CBOE gives the current chain only: the date of the `CalibrationID` should be the current one, and a
different one gets a warning. With `dir`, the chain is also saved there, as [`cboe_snapshot`](@ref) does, so that
`CboeInputStrategy(dir = dir)` reads it again later:

```julia
id = CalibrationID("SPX", Dates.format(today(), "dduyyyy"), "USD")
und, mkt = adapter(retriever(id, CboeWebInputStrategy(dir = "market_data")))
```
"""
struct CboeWebInputStrategy{R <: Real, M <: AbstractVector{<:Real}} <: InputType
    rate::R
    moneyness::M
    root::Union{Nothing, String}
    index::Bool
    dir::Union{Nothing, String}
end
CboeWebInputStrategy(; rate::Real = 0.04, moneyness::AbstractVector{<:Real} = 0.8:0.05:1.2, root::Union{Nothing, String} = nothing, index::Bool = true, dir::Union{Nothing, AbstractString} = nothing) = CboeWebInputStrategy(rate, moneyness, root, index, dir === nothing ? nothing : String(dir))

cboe_file(underlying, refDate::AbstractString, dir) = joinpath(dir, "CboeOptions_" * underlying * refDate * ".json")

# The current option chain of `underlying`, as the CBOE returns it, and its date as in a CalibrationID.
function cboe_download(underlying::AbstractString; index::Bool = true)
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/" * (index ? "_" : "") * underlying * ".json"
    body = String(take!(Downloads.download(url, IOBuffer(); headers = ["User-Agent" => "FinancialCalibrator.jl"])))
    return body, Dates.format(cboe_reference_date(JSON.parse(body)), "dduyyyy")
end

function save_chain(body, underlying, refDate, dir)
    mkpath(dir)
    return write(cboe_file(underlying, refDate, dir), body)
end

"""
    cboe_snapshot(underlying; index = true, currency = "USD", dir = pwd()) -> CalibrationID

Download the option chain of `underlying` from the CBOE delayed quotes (an index, such as `"SPX"`, is requested as
`_SPX`), save it as it is returned in `dir`, named after its date, and return the `CalibrationID` that reads it with
[`CboeInputStrategy`](@ref) from the same directory:

```julia
id = cboe_snapshot("SPX"; dir = "market_data")
und, mkt = adapter(retriever(id, CboeInputStrategy(dir = "market_data")))
```
"""
function cboe_snapshot(underlying::AbstractString; index::Bool = true, currency::AbstractString = "USD", dir::AbstractString = pwd())
    body, refDate = cboe_download(underlying; index = index)
    save_chain(body, underlying, refDate, dir)
    return CalibrationID(String(underlying), refDate, String(currency))
end

cboe_reference_date(chain) = Date(chain["timestamp"][1:10])

# root, expiry, call flag and strike of an OCC symbol: SPX261218C05000000 is the call of the SPX root expiring on
# 18 December 2026 with strike 5000
function parse_occ(symbol::AbstractString)
    m = match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", symbol)
    m === nothing && throw(ArgumentError("not an OCC option symbol: $symbol"))
    return (root = String(m[1]), expiry = Date(m[2], dateformat"yymmdd") + Year(2000), isCall = m[3] == "C", strike = parse(Int, m[4]) / 1000)
end

function retriever(calId::CalibrationID, input::CboeInputStrategy)::RawMarketData
    chain = JSON.parse(read(cboe_file(calId.underlying, calId.refDate, input.dir), String))
    return cboe_raw_market_data(chain, something(input.root, calId.underlying), input.rate, input.moneyness)
end

function retriever(calId::CalibrationID, input::CboeWebInputStrategy)::RawMarketData
    body, refDate = cboe_download(calId.underlying; index = input.index)
    refDate == calId.refDate || @warn "The CBOE gives the current option chain only: it is of $refDate, not of $(calId.refDate)."
    input.dir === nothing || save_chain(body, calId.underlying, refDate, input.dir)
    return cboe_raw_market_data(JSON.parse(body), something(input.root, calId.underlying), input.rate, input.moneyness)
end

# The quotes of one expiry that pass the filters: (strike, call flag, mid price, implied volatility).
function cboe_quotes(options, root)
    by_expiry = Dict{Date, Vector{Any}}()
    for o in options
        s = parse_occ(o["option"])
        s.root == root || continue
        bid, ask, iv = o["bid"], o["ask"], o["iv"]
        (bid > 0 && ask >= bid && 1e-3 < iv < 5) || continue
        push!(get!(by_expiry, s.expiry, []), (strike = s.strike, isCall = s.isCall, mid = (bid + ask) / 2, iv = iv))
    end
    return by_expiry
end

# The forward of an expiry by put–call parity, the median over the strikes nearest to the spot that have both a call and
# a put, or nothing with fewer than two such strikes.
function parity_forward(quotes, S0, r, T; strikes = 6)
    calls = Dict(q.strike => q.mid for q in quotes if q.isCall)
    puts = Dict(q.strike => q.mid for q in quotes if !q.isCall)
    common = sort(collect(intersect(keys(calls), keys(puts))), by = K -> abs(K - S0))
    length(common) < 2 && return nothing
    forwards = [K + exp(r * T) * (calls[K] - puts[K]) for K in common[1:min(strikes, end)]]
    sort!(forwards)
    n = length(forwards)
    return isodd(n) ? forwards[(n+1)÷2] : (forwards[n÷2] + forwards[n÷2+1]) / 2
end

# Linear interpolation of (x, y), x increasing, at the points of grid, or nothing if grid is not within the range of x.
function interpolate_on(x, y, grid)
    (isempty(x) || first(grid) < first(x) || last(grid) > last(x)) && return nothing
    return map(grid) do g
        j = clamp(searchsortedlast(x, g), 1, length(x) - 1)
        t = (g - x[j]) / (x[j+1] - x[j])
        return (1 - t) * y[j] + t * y[j+1]
    end
end

function cboe_raw_market_data(chain, root, r, grid)
    refDate = cboe_reference_date(chain)
    S0 = Float64(chain["data"]["current_price"])
    ref = string(refDate)
    rates, dividends, vols = Dict{String, Any}[], Dict{String, Any}[], Dict{String, Any}[]
    for (expiry, quotes) in sort(collect(cboe_quotes(chain["data"]["options"], root)), by = first)
        T = Dates.value(expiry - refDate) / 365
        T > 0 || continue
        F = parity_forward(quotes, S0, r, T)
        (F === nothing || F <= 0) && continue
        # out of the money, a strike once (the call and the put of the forward strike would both qualify)
        otm = sort(unique(q -> q.strike, [q for q in quotes if q.isCall == (q.strike >= F)]), by = q -> q.strike)
        smile = interpolate_on([q.strike / S0 for q in otm], [q.iv for q in otm], grid)
        smile === nothing && continue
        maturity = string(expiry)
        push!(rates, Dict("referenceDate" => ref, "maturityDate" => maturity, "value" => r))
        push!(dividends, Dict("referenceDate" => ref, "maturityDate" => maturity, "value" => r - log(F / S0) / T))
        append!(vols, [Dict("referenceDate" => ref, "maturityDate" => maturity, "pointValue" => m, "pointType" => "MONEYNESS", "value" => v) for (m, v) in zip(grid, smile)])
    end
    # the adapter interpolates the rates and the dividends with cubic splines
    length(rates) >= 4 || throw(ArgumentError("$(length(rates)) expiries of root $root cover the moneyness grid $(first(grid)) to $(last(grid)): at least 4 are needed"))
    spot = [Dict("referenceDate" => ref, "value" => S0)]
    return RawMarketData(refDate, JSON.json(spot), JSON.json(rates), JSON.json(dividends), JSON.json(vols))
end
