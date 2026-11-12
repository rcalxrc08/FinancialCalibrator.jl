function general_adapter_strategy(SpotString::String, RfString::String, IdString::String, VolaString::String, YfrTreshold::Tuple{Real, Real} = (0.1, 5.0), MoneynessTreshold::Tuple{Real, Real} = (0.7, 2.0), BasisConvention::Integer = 1)
    ####Input Parser and Adapter

    # JSON.parse - string or stream to Julia data structures
    jSpot = JSON.parse(SpotString)[1]
    jRf = JSON.parse(RfString)
    jImplDiv = JSON.parse(IdString)
    jVolat = JSON.parse(VolaString)

    ######## Parsing of Input Data
    #BasisConvention=1;#ACT/365
    #Spot Parsing
    S0=jSpot["value"]
    RefDate=Date(jSpot["referenceDate"])

    #Zero Rate Parsing
    RfArray=[jRf[i]["value"] for i = 1:length(jRf)]
    DatesRfArray=Date.([jRf[i]["maturityDate"] for i = 1:length(jRf)])
    yfrRfArray=[yearfrac(RefDate, matDate, BasisConvention) for matDate in DatesRfArray]
    IndexRf=sortperm(yfrRfArray)
    yfrRfArray=yfrRfArray[IndexRf]
    RfArray=RfArray[IndexRf]

    #Implied Dividend Parsing
    IdArray=[jImplDiv[i]["value"] for i = 1:length(jImplDiv)]
    DatesIdArray=Date.([jImplDiv[i]["maturityDate"] for i = 1:length(jImplDiv)])
    yfrIdArray=[yearfrac(RefDate, matDate, BasisConvention) for matDate in DatesIdArray]
    IndexId=sortperm(yfrIdArray)
    yfrIdArray=yfrIdArray[IndexId]
    IdArray=IdArray[IndexId]

    #Implied Vol Parsing
    maturityDates=((sort(unique([jVolat[i]["maturityDate"] for i = 1:length(jVolat)]))))
    moneyness=sort(unique([jVolat[i]["pointValue"] for i = 1:length(jVolat)]))
    VolaMatrix=Matrix(undef, length(maturityDates), length(moneyness))

    for i = 1:length(maturityDates)
        volaTmp=([jVolat[j]["value"] for j = 1:length(jVolat) if jVolat[j]["maturityDate"]==maturityDates[i]])
        moneyTemp=([jVolat[j]["pointValue"] for j = 1:length(jVolat) if jVolat[j]["maturityDate"]==maturityDates[i]])
        IndexMoney=sortperm(moneyTemp)
        VolaMatrix[i, 1:end]=volaTmp[IndexMoney]
    end

    maturityDates=Date.(maturityDates)

    yfrVolaArray=[yearfrac(RefDate, matDate, BasisConvention) for matDate in maturityDates]

    ##### Augmenting Phase

    #Augment ZeroRate
    ZeroSplineInterpolator=Dierckx.Spline1D(yfrRfArray, RfArray)
    ZeroRateAug=ZeroSplineInterpolator(yfrVolaArray)

    #Augment ZeroRate
    IdSplineInterpolator=Dierckx.Spline1D(yfrIdArray, IdArray)
    IdAug=IdSplineInterpolator(yfrVolaArray)

    ##### Resize All the Inputs
    # The maturities and the moneyness points within the thresholds, and the same rows and columns of the volatilities:
    # the rows of VolaMatrix are all the maturities, so they are selected with the mask of the maturities (#2).
    rows = findall(t -> YfrTreshold[1] < t < YfrTreshold[2], yfrVolaArray)
    columns = findall(m -> MoneynessTreshold[1] < m < MoneynessTreshold[2], moneyness)
    TimeToMaturity = yfrVolaArray[rows]
    maturityDates = maturityDates[rows]
    Moneyness = moneyness[columns]
    ZeroRates = ZeroRateAug[rows]
    ImpliedDividend = IdAug[rows]
    Volatility = Matrix{Float64}(VolaMatrix[rows, columns])
    Strikes=Moneyness .* S0

    return (S0, TimeToMaturity, maturityDates, ZeroRates, ImpliedDividend, Strikes, Volatility)
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
