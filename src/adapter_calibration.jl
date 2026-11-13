# A curve given as JSON points (maturity date and value): year fractions from `refDate` and values, sorted by maturity.
function parse_curve(points, refDate::Date, basis::Integer)
    t = [yearfrac(refDate, Date(p["maturityDate"]), basis) for p in points]
    order = sortperm(t)
    return t[order], [Float64(p["value"]) for p in points][order]
end

# Linear interpolation of the nodes (x, y) (x increasing) at t, flat beyond the first and the last node.
function interpolate_linear(x::AbstractVector, y::AbstractVector, t::Real)
    t <= first(x) && return first(y)
    t >= last(x) && return last(y)
    i = searchsortedlast(x, t)
    w = (t - x[i]) / (x[i+1] - x[i])
    return (1 - w) * y[i] + w * y[i+1]
end

function general_adapter_strategy(SpotString::String, RfString::String, IdString::String, VolaString::String, YfrTreshold::Tuple{Real, Real} = (0.1, 5.0), MoneynessTreshold::Tuple{Real, Real} = (0.7, 2.0), BasisConvention::Integer = 1)
    spot = JSON.parse(SpotString)[1]
    S0 = Float64(spot["value"])
    refDate = Date(spot["referenceDate"])
    rate_times, rates = parse_curve(JSON.parse(RfString), refDate, BasisConvention)
    dividend_times, dividends = parse_curve(JSON.parse(IdString), refDate, BasisConvention)

    # The surface: one row per maturity, one column per moneyness, filled in a single pass over the points.
    points = JSON.parse(VolaString)
    maturities = sort!(unique(p["maturityDate"] for p in points))
    moneyness = sort!(unique(Float64(p["pointValue"]) for p in points))
    row = Dict(m => i for (i, m) in enumerate(maturities))
    column = Dict(m => j for (j, m) in enumerate(moneyness))
    vols = fill(NaN, length(maturities), length(moneyness))
    for p in points
        vols[row[p["maturityDate"]], column[Float64(p["pointValue"])]] = p["value"]
    end
    any(isnan, vols) && throw(ArgumentError("the volatility surface does not have a quote at every maturity and moneyness"))
    maturityDates = Date.(maturities)
    times = [yearfrac(refDate, d, BasisConvention) for d in maturityDates]

    # The maturities and the moneyness points within the thresholds, and the same rows and columns of the volatilities (#2).
    rows = findall(t -> YfrTreshold[1] < t < YfrTreshold[2], times)
    columns = findall(m -> MoneynessTreshold[1] < m < MoneynessTreshold[2], moneyness)
    TimeToMaturity = times[rows]
    # rates and dividends at the maturities of the surface: linear in the year fraction
    ZeroRates = [interpolate_linear(rate_times, rates, t) for t in TimeToMaturity]
    ImpliedDividend = [interpolate_linear(dividend_times, dividends, t) for t in TimeToMaturity]
    return (S0, TimeToMaturity, maturityDates[rows], ZeroRates, ImpliedDividend, moneyness[columns] .* S0, vols[rows, columns])
end

"""
    adapter(raw::RawMarketData, T = Float64) -> (underlying, mkt)

Parse the raw market data. Returns the `Underlying` (spot and implied dividend curve over the maturities) and the
`MarketData` (rates, strikes and volatilities), both with floating point type `T`.
"""
function adapter(jsonMktDataIN::RawMarketData, ::Type{T} = Float64) where {T <: AbstractFloat}
    ####Input Adapter
    adaptedData=general_adapter_strategy(jsonMktDataIN.SpotString, jsonMktDataIN.RfString, jsonMktDataIN.ImplDivString, jsonMktDataIN.VolatString)
    S0, TimeToMaturity, maturityDates, ZeroRates, ImpliedDividend, Strikes, Volatility = adaptedData

    dividend_curve = FinancialMonteCarlo.ImpliedCurve(Vector{T}(ImpliedDividend), Vector{T}(TimeToMaturity))
    underlying = FinancialMonteCarlo.Underlying(T(S0), dividend_curve)
    mkt = MarketData(Vector{T}(TimeToMaturity), Vector{Date}(maturityDates), Vector{T}(ZeroRates), Vector{T}(Strikes), Matrix{T}(Volatility))
    return underlying, mkt
end
