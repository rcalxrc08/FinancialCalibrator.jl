
abstract type InputType end

struct IOFileInputStrategy <: InputType end

struct WebFileInput <: InputType end

struct CalibrationID
    underlying::String
    refDate::String
    currency::String
    CalibrationID(underlyingIn, refDateIn, currencyIn) = new(underlyingIn, refDateIn, currencyIn)
end

struct RawMarketData
    refDate::Date
    SpotString::String
    RfString::String
    ImplDivString::String
    VolatString::String
    RawMarketData(refDate, SpotStringIn, RfStringIn, ImplDivStringIn, VolatStringIn) = new(refDate, SpotStringIn, RfStringIn, ImplDivStringIn, VolatStringIn)
end

function io_retriever(SpotRelPath::String, RfRelPath::String, IdRelPath::String, VolaRelPath::String)
    ####Input Retriever
    SpotString=read(SpotRelPath, String)
    RfString=read(RfRelPath, String)
    ImplDivString=read(IdRelPath, String)
    VolatString=read(VolaRelPath, String)

    return (SpotString, RfString, ImplDivString, VolatString)
end

function web_retriever(SpotRelPath::String, RfRelPath::String, IdRelPath::String, VolaRelPath::String)
    ####Input Retriever
    SpotString=readdlm(download(SpotRelPath))[1]
    if (SpotString=="[]")
        error("Data Not Available")
    end
    RfString=readdlm(download(RfRelPath))[1]
    ImplDivString=readdlm(download(IdRelPath))[1]
    VolatString=readdlm(download(VolaRelPath))[1]

    return (SpotString, RfString, ImplDivString, VolatString)
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
