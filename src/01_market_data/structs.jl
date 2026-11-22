
abstract type InputType end

struct IOFileInputStrategy <: InputType end

"""
    CalibrationID(underlying, refDate, currency)
    CalibrationID(; underlying, refDate, currency)

The market data of a calibration: the `underlying` (`"Dax"`, `"SPX"`), the reference date `refDate`, a `Date` or a
string with day, month and year as in `"22Aug2016"` (the names of the data files), and the ISO 4217 code of the
`currency` (`"EUR"`). The three are checked here (#6): a date that does not parse or a currency that is not three capital
letters throws an `ArgumentError`, so that swapped arguments fail at once rather than as a missing file later.
"""
struct CalibrationID
    underlying::String
    refDate::String
    currency::String
    function CalibrationID(underlying::AbstractString, refDate::AbstractString, currency::AbstractString)
        parse_ref_date(refDate)   # throws an ArgumentError with the expected format
        occursin(r"^[A-Z]{3}$", currency) || throw(ArgumentError("the currency of a CalibrationID is an ISO 4217 code such as \"EUR\", not \"$currency\""))
        return new(String(underlying), String(refDate), String(currency))
    end
end
CalibrationID(underlying::AbstractString, refDate::Date, currency::AbstractString) = CalibrationID(underlying, Dates.format(refDate, REF_DATE_FORMAT), currency)
CalibrationID(; underlying::AbstractString, refDate::Union{AbstractString, Date}, currency::AbstractString) = CalibrationID(underlying, refDate, currency)

# The raw data and the day count `basis` (FinancialToolbox's `yearfrac`) its times are measured in: the adapter reads the
# curves and the surface with the same basis that built them (#28). 1 (30/360 SIA) for the files of data/json, 3
# (ACT/365) for the CBOE chains.
struct RawMarketData
    refDate::Date
    SpotString::String
    RfString::String
    ImplDivString::String
    VolatString::String
    basis::Int
    RawMarketData(refDate, SpotStringIn, RfStringIn, ImplDivStringIn, VolatStringIn, basis::Integer = 1) = new(refDate, SpotStringIn, RfStringIn, ImplDivStringIn, VolatStringIn, basis)
end

"""
    MarketData

Market quotes of a calibration: maturities, zero rates, strikes and the implied volatility surface
(`Volatility[i, j]` is the volatility of maturity `i` and strike `j`). Spot and dividends are not part of it: they belong
to the `Underlying` that `adapter` returns next to it, which the model to calibrate is built on.
`Strikes` and `Volatility` can live on any array type, e.g. on the GPU: `Adapt.adapt(CuArray, mkt)` moves them to the
device (the per-maturity vectors stay on the host).
"""
struct MarketData{T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    TimeToMaturity::Vector{T}
    MaturityDates::Vector{Date}
    ZeroRates::Vector{T}
    Strikes::V
    Volatility::M
end

# The floating point type is the promotion of the types of all the numeric inputs.
function MarketData(TimeToMaturity::AbstractVector{<:Real}, MaturityDates::AbstractVector{Date}, ZeroRates::AbstractVector{<:Real}, Strikes::AbstractVector{<:Real}, Volatility::AbstractMatrix{<:Real})
    T = float(promote_type(eltype(TimeToMaturity), eltype(ZeroRates), eltype(Strikes), eltype(Volatility)))
    return MarketData(Vector{T}(TimeToMaturity), Vector{Date}(MaturityDates), Vector{T}(ZeroRates), Vector{T}(Strikes), Matrix{T}(Volatility))
end

Base.eltype(::Type{<:MarketData{T}}) where {T} = T
Base.eltype(::MarketData{T}) where {T} = T

# Dividend yield of `underlying`, flat-equivalent over the maturity `mat`: it is the one the pricers use, so that the
# prices are inverted consistently.
implied_dividend(underlying::FinancialMonteCarlo.AbstractUnderlying, mat::Real) = FinancialMonteCarlo.integral(underlying.d, mat) / mat

Adapt.adapt_structure(to, m::MarketData) = MarketData(m.TimeToMaturity, m.MaturityDates, m.ZeroRates, Adapt.adapt(to, m.Strikes), Adapt.adapt(to, m.Volatility))
