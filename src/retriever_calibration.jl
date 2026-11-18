"""
    retriever(id::CalibrationID, input) -> RawMarketData

The raw market data of `id` (`CalibrationID(underlying, refDate, currency)`, with `refDate` as `22Aug2016`), read by
the input strategy `input`: `IOFileInputStrategy()` reads the JSON files of `data/json` (spot, zero rates, implied
dividends and implied volatilities), [`CboeInputStrategy`](@ref) an option chain of the CBOE saved on disk, and [`CboeWebInputStrategy`](@ref) the current one,
downloaded. [`adapter`](@ref) turns
the result into the `Underlying` and the [`MarketData`](@ref) of a calibration.
"""
function retriever(calId::CalibrationID, ::IOFileInputStrategy)
    (; underlying, refDate, currency) = calId
    refDay = parse_ref_date(refDate)   # first: a malformed date fails here, not as a missing file
    # the files of data/json are named <kind><underlying or currency><refDate>.json, e.g. SpotDax22Aug2016.json
    read_json(kind, name) = read(joinpath(DATA_DIR, kind * name * refDate * ".json"), String)
    return RawMarketData(refDay, read_json("Spot", underlying), read_json("Rf", currency), read_json("ImplDiv", underlying), read_json("Volatility", underlying))
end

# The reference date of a CalibrationID, `22Aug2016`: the format of the names of the data files, and the one the CBOE
# strategies write the date of a downloaded chain in.
const REF_DATE_FORMAT = dateformat"dduuuyyyy"   # uuu: a fixed-width month, which "dduyyyy" cannot parse

function parse_ref_date(refDate::AbstractString)
    try
        return Date(refDate, REF_DATE_FORMAT)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("the reference date of a CalibrationID is day, month and year as in \"22Aug2016\", not \"$refDate\""))
    end
end
