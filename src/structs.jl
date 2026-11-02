
abstract type InputType end

struct IOFileInputStrategy <: InputType end
struct IOFileInputMDS <: InputType end

struct WebFileInput <: InputType end
struct WebFileInputMDS <: InputType end

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

struct RawMarketDataMDS
    refDate::Date
    SpotString::String
    RfString::String
    ImplDivString::String
    VolatString::String
    RawMarketDataMDS(refDate, SpotStringIn, RfStringIn, ImplDivStringIn, VolatStringIn) = new(refDate, SpotStringIn, RfStringIn, ImplDivStringIn, VolatStringIn)
end

function IORetriever(SpotRelPath::String, RfRelPath::String, IdRelPath::String, VolaRelPath::String)
    ####Input Retriever
    SpotString=read(SpotRelPath, String)
    RfString=read(RfRelPath, String)
    ImplDivString=read(IdRelPath, String)
    VolatString=read(VolaRelPath, String)

    return (SpotString, RfString, ImplDivString, VolatString)
end

function WEBRetriever(SpotRelPath::String, RfRelPath::String, IdRelPath::String, VolaRelPath::String)
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

Market data of a calibration. `Strikes` and `Volatility` can live on any array type, e.g. on the GPU:
`Adapt.adapt(CuArray, mkt)` moves them to the device (the per-maturity vectors stay on the host).
"""
struct MarketData{T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    S0::T
    TimeToMaturity::Vector{T}
    MaturityDates::Vector{Date}
    ZeroRates::Vector{T}
    ImpliedDividend::Vector{T}
    Strikes::V
    Volatility::M
end

# The floating point type is the promotion of the types of all the numeric inputs.
function MarketData(S0::Real, TimeToMaturity::AbstractVector{<:Real}, MaturityDates::AbstractVector{Date}, ZeroRates::AbstractVector{<:Real}, ImpliedDividend::AbstractVector{<:Real}, Strikes::AbstractVector{<:Real}, Volatility::AbstractMatrix{<:Real})
    T = float(promote_type(typeof(S0), eltype(TimeToMaturity), eltype(ZeroRates), eltype(ImpliedDividend), eltype(Strikes), eltype(Volatility)))
    return MarketData(T(S0), Vector{T}(TimeToMaturity), Vector{Date}(MaturityDates), Vector{T}(ZeroRates), Vector{T}(ImpliedDividend), Vector{T}(Strikes), Matrix{T}(Volatility))
end

Base.eltype(::Type{<:MarketData{T}}) where {T} = T
Base.eltype(::MarketData{T}) where {T} = T

Adapt.adapt_structure(to, m::MarketData) = MarketData(m.S0, m.TimeToMaturity, m.MaturityDates, m.ZeroRates, m.ImpliedDividend, Adapt.adapt(to, m.Strikes), Adapt.adapt(to, m.Volatility))
