"""
    retriever(id::CalibrationID, input) -> RawMarketData

The raw market data of `id` (`CalibrationID(underlying, refDate, currency)`, with `refDate` as `22Aug2016`), read by
the input strategy `input`: `IOFileInputStrategy()` reads the JSON files of `data/json` (spot, zero rates, implied
dividends and implied volatilities), [`CboeInputStrategy`](@ref) an option chain of the CBOE saved on disk, and [`CboeWebInputStrategy`](@ref) the current one,
downloaded. [`adapter`](@ref) turns
the result into the `Underlying` and the [`MarketData`](@ref) of a calibration.
"""
function retriever(calId::CalibrationID, ioInput::IOFileInputStrategy)::RawMarketData
    ####Input Retriever
    currency=calId.currency
    underlying=calId.underlying
    refDate=calId.refDate
    path=DATA_DIR
    SpotRelPath=joinpath(path, "Spot"*underlying*refDate*".json")
    RfRelPath=joinpath(path, "Rf" * currency * refDate * ".json")
    ImplDivRelPath=joinpath(path, "ImplDiv"*underlying*refDate*".json")
    VolaRelPath = joinpath(path, "Volatility"*underlying*refDate*".json")
    TupleString=io_retriever(SpotRelPath, RfRelPath, ImplDivRelPath, VolaRelPath)

    adjRefDate=refDate[1:2]*"-"*refDate[3:5]*"-"*refDate[6:end]

    MyDateFormat=DateFormat("d-u-y")
    RefDate=Date(adjRefDate, MyDateFormat)
    MyDateFormat=DateFormat("d-u-y")
    RefDate=Date(adjRefDate, MyDateFormat)
    rawMarketData=RawMarketData(RefDate, TupleString[1], TupleString[2], TupleString[3], TupleString[4])
    return rawMarketData
end
