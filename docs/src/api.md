```@raw html
<div class="fc-page-head">
<p class="fc-kicker"><span>Reference</span> Every exported name</p>
```

# API reference

```@raw html
<p class="fc-page-lede">Four verbs: <strong>read</strong> market data, <strong>calibrate</strong> a model,
<strong>differentiate</strong> the calibration, <strong>price</strong> with a closed form. The models and the
numerical methods come from FinancialMonteCarlo.jl and FinancialFFT.jl.</p>
</div>
<div class="fc-grid fc-grid-3">
<div class="fc-card"><span class="fc-card-k is-blue">Calibrate</span><p><a href="#FinancialCalibrator.calibrate"><code>calibrate</code></a> fit a model to a surface<br><a href="#FinancialCalibrator.Parametrization"><code>Parametrization</code></a> the variables of the search<br><a href="#FinancialCalibrator.default_parametrization"><code>default_parametrization</code></a> one per model<br><a href="#FinancialCalibrator.CalibratedData"><code>CalibratedData</code></a> the result</p></div>
<div class="fc-card"><span class="fc-card-k">Differentiate</span><p><a href="#FinancialCalibrator.calibrated_parameters"><code>calibrated_parameters</code></a> p*(v), differentiable<br><a href="#FinancialCalibrator.vol_errors"><code>vol_errors</code></a> the residuals<br><a href="#FinancialCalibrator.loss"><code>loss</code></a> half their sum of squares</p></div>
<div class="fc-card"><span class="fc-card-k is-amber">Market data</span><p><a href="#FinancialCalibrator.MarketData"><code>MarketData</code></a> the surface<br><a href="#FinancialCalibrator.retriever"><code>retriever</code></a>, <a href="#FinancialCalibrator.adapter"><code>adapter</code></a> read and parse<br><a href="#FinancialCalibrator.CboeInputStrategy"><code>CboeInputStrategy</code></a>, <a href="#FinancialCalibrator.CboeWebInputStrategy"><code>CboeWebInputStrategy</code></a>, <a href="#FinancialCalibrator.cboe_snapshot"><code>cboe_snapshot</code></a> CBOE quotes</p></div>
<div class="fc-card"><span class="fc-card-k is-blue">Pricing</span><p><a href="#FinancialCalibrator.AnalyticMethod"><code>AnalyticMethod</code></a> closed forms: Black–Scholes, shifted lognormal mixture</p></div>
</div>
```

## Calibration

```@docs
calibrate
Parametrization
default_parametrization
CalibratedData
```

## Derivatives of the calibration

```@docs
calibrated_parameters
vol_errors
FinancialCalibrator.loss
```

## Market data

[`retriever`](@ref) reads the raw data of a calibration and [`adapter`](@ref) turns it into the `Underlying` of the
models and the [`MarketData`](@ref). Besides the bundled DAX surface, [`CboeInputStrategy`](@ref) reads a saved CBOE
option chain, implies the forward and the dividends from put–call parity, and resamples the out of the money
volatilities on a moneyness grid; [`CboeWebInputStrategy`](@ref) downloads the current chain and can save it:

```julia
id = CalibrationID("SPX", Dates.format(today(), "dduyyyy"), "USD")
und, mkt = adapter(retriever(id, CboeWebInputStrategy(dir = "market_data")))   # downloaded, and saved in market_data/
und, mkt = adapter(retriever(id, CboeInputStrategy(dir = "market_data")))      # read again from the saved chain
```

```@docs
MarketData
retriever
adapter
CboeInputStrategy
CboeWebInputStrategy
cboe_snapshot
```

## Pricing

```@docs
AnalyticMethod
```

## Index

```@index
```
