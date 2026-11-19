# Stage 01: market data

Quotes in, `(underlying, mkt::MarketData)` out.

## Inputs

| Load | Why |
|:--|:--|
| `structs.jl` | `CalibrationID`, input strategies, `RawMarketData`, `MarketData{T}` (maturities, rates, strikes, vols only), `implied_dividend` |
| `retriever_calibration.jl` | reads `data/json/`; `REF_DATE_FORMAT` and `parse_ref_date` |
| `cboe.jl` | CBOE chains: `CboeInputStrategy`, `CboeWebInputStrategy`, `cboe_snapshot` |
| `adapter_calibration.jl` | JSON → surface, `interpolate_linear`, the `ImpliedCurve` of the dividends |
| `references/market_data.md` | formats, filters, parity, the sample data, the snapshots folder |
| `references/public_api.md` | where spot and dividends live |

## Process

- A new source is a new input strategy plus a `retriever` method producing the JSON strings of the existing schema, so
  `adapter` is reused.
- The adapter selects rows and columns of the surface with the same masks as the maturities and moneyness (#2).
- No interpolation package: linear in the year fraction, flat outside the nodes.

## Outputs

`adapter(raw, T)` returns the `Underlying` (spot, implied dividend curve: a FinancialMonteCarlo `Curve` of `SVector`s,
isbits since FinancialMonteCarlo 0.9) and `MarketData{T}`. Checked by the core testset of `test/runtests.jl` and
`test/cboe.jl`.
