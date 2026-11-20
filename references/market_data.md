# Market data: files, CBOE, adapter

_Stage 01 reference._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

Files of `src/01_market_data/` (included in this order): `structs.jl` (input types, `MarketData`,
`implied_dividend`), `retriever_calibration.jl` (reads `data/json/`; the reference date `22Aug2016` is parsed and written with `REF_DATE_FORMAT = dateformat"dduuuyyyy"`, shared with `cboe.jl`: `"dduyyyy"` formats the same string but cannot parse it, the month has no fixed width; `parse_ref_date` throws an `ArgumentError` naming the expected format, #12), `cboe.jl` (`CboeInputStrategy` from a file, `CboeWebInputStrategy` downloading the current chain, `cboe_snapshot`:
CBOE delayed quotes, #18), `adapter_calibration.jl` (parsing, returns
`(underlying, mkt)`; the surface is filled in one pass over the JSON points, rates and dividends are interpolated
linearly in the year fraction, flat outside the nodes, by `interpolate_linear`: no interpolation package, Dierckx was
dropped; its cubic spline oscillated up to 2.5 bp between the nodes of the DAX rate curve).

`src/01_market_data/cboe.jl`: `CboeWebInputStrategy` and `cboe_snapshot` download (`cboe_download`) `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` with the
`Downloads` stdlib (no HTTP.jl) and saves it (in `pwd()` by default: the package directory is read-only once installed) as `CboeOptions_<underlying><ddMonyyyy>.json`; `retriever(id,
CboeInputStrategy(; rate, moneyness, root, dir))` turns a saved chain into the JSON strings of the existing schema, so
`adapter` is reused: OCC symbols parsed by `parse_occ`, quotes with `bid > 0` and `1e-3 < iv < 5`, the forward of each
expiry by put–call parity (median over the 6 strikes nearest to the spot), the dividend `r - log(F/S0)/T`, a flat
`rate`, out of the money vols interpolated linearly in moneyness on the grid (expiries not covering it are dropped,
since the adapter needs one grid for all maturities, #11). At least one expiry. Yahoo's option
chain needs a cookie and crumb that it refuses to non-browser clients (401/429), so it is not implemented. No real chain is in the repository (too heavy): `test/cboe.jl` builds a synthetic chain
in the CBOE format (Black–Scholes with a skew, bid/ask 1% around the price, so parity is exact) in a temporary
directory; `test/local/cboe_spx.jl` (NIG calibrated to the real SPX chain and its implicit derivative against
recalibration) runs only when `data/json/cboe/2026-11-12/CboeOptions_SPX12Nov2026.json` exists. Snapshots are git-ignored (the
chain is 14 MB): CBOE chain and spot and FRED Treasury curve in `data/json/cboe/<yyyy-mm-dd>/`, ECB and Yahoo in
`data/json/real/`, whose README describes all the sources. The adapter now selects the volatility rows with the same mask as the maturities (#2).

- The sample data is `data/json/*Dax22Aug2016.json`.
- `ImpliedCurve` prepends the point `0 => 0` so the integrated dividend over a maturity is not the spot value at that maturity.
