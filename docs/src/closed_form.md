```@raw html
<div class="fc-page-head">
<p class="fc-kicker"><span>Guide 03</span> Exact prices, no numerical method</p>
```

# Closed forms: the shifted lognormal mixture

```@raw html
<p class="fc-page-lede"><a href="../api/#FinancialCalibrator.AnalyticMethod"><code>AnalyticMethod</code></a> prices
a smile with the closed form of the model, when it has one: Black and Scholes, and the <strong>shifted lognormal
mixture</strong> of Brigo and Mercurio. The second is a model worth calibrating on its own, and this page is about
it.</p>
<div class="fc-meta"><span>Brigo–Mercurio</span><span>AnalyticMethod</span><span>arbitrage free</span><span>GPU</span><span class="is-live">computed at build time</span></div>
<nav class="fc-toc">
<a href="#The-model"><b>01</b>The model</a>
<a href="#Why-calibrate-it"><b>02</b>Why calibrate it</a>
<a href="#On-the-DAX-surface"><b>03</b>On the DAX surface</a>
<a href="#Other-closed-forms"><b>04</b>Other closed forms</a>
</nav>
</div>
```

## The model

The spot is a deterministic shift plus a process ``A`` whose law at each time is a mixture of lognormals:

```math
S_t = \alpha\, S_0\, e^{(r - d) t} + A_t, \qquad A_0 = (1 - \alpha) S_0,
\qquad A_T \sim \sum_{j=1}^{n} w_j\, A_0\, e^{(r - d - \eta_j^2 / 2) T + \eta_j \sqrt{T} Z},
```

with ``Z`` standard normal, volatilities ``\eta_1, \dots, \eta_n > 0`` and weights ``w_j = \lambda_j`` for
``j < n``, ``w_n = 1 - \sum_{j<n} \lambda_j``. `ShiftedLogNormalMixture(η, λ, α, underlying)` takes the ``n``
volatilities, the ``n - 1`` free weights and the shift ``\alpha < 1``. Brigo and Mercurio show that a local volatility
``\sigma(t, A_t)`` exists for which ``A`` has exactly these marginals: the model is a diffusion, and not only a family
of densities. FinancialMonteCarlo's simulation of `ShiftedLogNormalMixture` is not that diffusion (it sums the
lognormal processes path by path, a model with other European prices, see
[FinancialMonteCarlo.jl#56](https://github.com/rcalxrc08/FinancialMonteCarlo.jl/issues/56)): do not price exotics by
Monte Carlo with a model calibrated by its closed form until that is settled.

A European option on ``S`` is an option on ``A`` struck at ``\kappa = K - \alpha S_0 e^{(r - d) T}``, and the mixture
passes through the expectation: its price is the mixture of Black prices,

```math
C(K, T) = \sum_{j=1}^{n} w_j\, C_{\text{black}}\!\left((1 - \alpha) F_T, \kappa, r, T, \eta_j\right), \qquad
F_T = S_0 e^{(r - d) T},
```

and the same for a put. For ``\kappa \le 0`` (a positive shift and a strike below it) the call is always exercised,
``C = e^{-rT}(F_T - K)``, and the put is worthless. The implementation computes the shift, the forward and the weights
once per smile, then one Black price per component and strike, by `blkprice` of FinancialToolbox.

## Why calibrate it

```@raw html
<div class="fc-grid fc-grid-3">
<div class="fc-card"><span class="fc-card-k is-blue">Exact and fast</span><p>A price is <i>n</i> Black formulas: no quadrature, no truncation range, no simulation. A smile of a few hundred strikes with three components prices in about 0.15 ms on a laptop. The loss has no numerical noise, so the optimizer and the Levenberg–Marquardt polish reach a stationary point to machine precision, and the implicit derivatives of <code>calibrated_parameters</code> are those of the exact calibration.</p></div>
<div class="fc-card"><span class="fc-card-k">Free of static arbitrage</span><p>A mixture of arbitrage-free prices is arbitrage free: the prices are decreasing and convex in the strike, and the forward is matched exactly, for every parameter, so no constraint is needed beyond the signs of η and λ.</p></div>
<div class="fc-card"><span class="fc-card-k is-amber">Intuitive parameters</span><p>The mixture gives fat tails, the smile: the volatilities η are the regimes, the weights how likely they are. The shift gives the skew: α &lt; 0 makes the spot more lognormal than <i>A</i>, with a downward sloping implied volatility; α &gt; 0 the opposite.</p></div>
<div class="fc-card"><span class="fc-card-k is-blue">Differentiable by every backend</span><p>The formula is a broadcast of <code>blkprice</code>, which has closed-form ChainRules rules: ForwardDiff, Zygote and Mooncake differentiate it as well as the Fourier methods.</p></div>
<div class="fc-card"><span class="fc-card-k">On the GPU</span><p>The components are a tuple whose length is a type parameter, so a smile is one broadcast over its strikes, one thread per strike. With the strikes of <code>MarketData</code> on the device, or <code>AnalyticMethod(; mode = CudaMode())</code>, the prices never leave it.</p></div>
</div>
```
Its limit is the term structure. The parameters are the same at every maturity, and the smile of a mixture of
lognormals flattens with the maturity as ``1/\sqrt{T}`` in log-moneyness: one set of parameters fits a few maturities,
or one maturity at a time, better than a whole surface. Its local volatility dynamics also give forward smiles that are
flatter than the market's, which matters for forward starting and cliquet options, not for the European quotes it is
calibrated to.

## On the DAX surface

The DAX surface of 22 August 2016, with three components, against Black and Scholes (a single component and no shift),
both with the closed form:

```@example closed_form
using FinancialCalibrator, FinancialMonteCarlo

underlying, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
rms(res) = sqrt(2 * res.OptimalValue / count(!isnan, res.ModelVolatility))

bs = calibrate(BlackScholesProcess(0.2, underlying), AnalyticMethod(), mkt)
slnm = calibrate(ShiftedLogNormalMixture([0.1, 0.2, 0.35], [0.3, 0.4], -0.1, underlying), AnalyticMethod(), mkt)
(black_scholes = rms(bs), shifted_lognormal_mixture = rms(slnm))
```

```@example closed_form
Main.DocumentationFigures.closed_form_tiles(bs, slnm)  # hide
```

Four of the nine maturities, the model volatilities at the quotes:

![Black–Scholes and the shifted lognormal mixture against the DAX quotes](assets/figures/closed-form-light.svg)![Black–Scholes and the shifted lognormal mixture against the DAX quotes](assets/figures/closed-form-dark.svg)

The calibrated parameters, ``(\eta_1, \eta_2, \eta_3, \lambda_1, \lambda_2, \alpha)``:

```@example closed_form
slnm.OptimalParam
```

## Other closed forms

A model gets a closed form by two methods, `FinancialCalibrator.analytic_price_type(model)` (the number type of its
prices) and `FinancialCalibrator.analytic_kernel(model, S0, r, d, T)` (a callable `(K, isCall) -> price`, an isbits
struct so that a GPU broadcast captures it); `pricer(model, zero_rate, AnalyticMethod(), smile)` does the rest, on the
host or on the device.
