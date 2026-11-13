```@raw html
<div class="fc-hero">
<div class="fc-hero-text">
<p class="fc-eyebrow"><img src="assets/logo.png" alt="" width="28" height="28"> FinancialCalibrator.jl</p>
```

# Differentiable Calibration of Option Pricing Models

```@raw html
<p class="fc-lede">Calibrate volatility surfaces, compute sensitivities, and obtain quote-to-risk derivatives through
automatic differentiation in Julia.</p>
<nav class="fc-cta">
<a class="fc-btn fc-btn-primary" href="#Quick-start">Get Started <span aria-hidden="true">→</span></a>
<a class="fc-btn" href="https://github.com/rcalxrc08/FinancialCalibrator.jl"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg> View on GitHub</a>
</nav>
<pre class="fc-hero-code"><code><span class="k">res</span>   = calibrate(model, method, mkt)            <span class="c"># p*</span>
<span class="k">p(v)</span>  = calibrated_parameters(model, method, mkt, v)
<span class="k">∂p_∂v</span> = ForwardDiff.jacobian(p, mkt.Volatility) <span class="c"># ∂p*/∂v</span></code></pre>
</div>
<figure class="fc-hero-figure">
<img src="assets/figures/hero-surface-dark.svg" alt="Implied volatility surface of a Heston model calibrated to the DAX, with the market quotes as dots">
<img src="assets/figures/hero-surface-light.svg" alt="Implied volatility surface of a Heston model calibrated to the DAX, with the market quotes as dots">
<figcaption><span class="fc-legend"><i class="fc-dot"></i>DAX quote <i class="fc-dot fc-dot-hollow"></i>left out of the loss (low vega) <i class="fc-ramp"></i>calibrated Heston surface, coloured by implied volatility</span>
Heston calibrated to the DAX surface of 22 August 2016, rendered from the package's output when this site was built.</figcaption>
</figure>
</div>
```

```@example home
Main.DocumentationFigures.hero_stats()  # hide
```

## What it does

```@raw html
<p class="fc-section-lede">Most calibrators return a point estimate. FinancialCalibrator.jl returns the calibrated
parameters <em>as a differentiable function of the quotes</em>: the derivatives come from the implicit function theorem
at the optimum, never from unrolling the optimizer, so every risk number downstream of the calibration is one
gradient away.</p>
<div class="fc-grid fc-features">
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M3 24 C9 24 10 8 16 8 S23 20 29 12"/><circle cx="16" cy="8" r="2.4"/></svg></div>
<h3>Differentiable calibration</h3><p><code>calibrated_parameters</code> is the calibration as a function of the quoted volatilities, with its own derivative rules.</p><p class="fc-why">Sensitivities without bump-and-recalibrate: one call instead of two calibrations per quote.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="12.5" y="4" width="7" height="7" rx="1.5"/><rect x="21" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="12.5" width="7" height="7" rx="1.5"/><rect x="12.5" y="12.5" width="7" height="7" rx="1.5" class="hl"/><rect x="21" y="12.5" width="7" height="7" rx="1.5"/><path d="M16 21v7"/></svg></div>
<h3>Quote-to-risk sensitivities</h3><p>The derivative of any contract priced with the calibrated model, an exotic or a whole position, with respect to every quote of the surface.</p><p class="fc-why">Bucketed vega that is consistent with the model you price with, from a single gradient.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M4 9h24M4 16h24M4 23h24"/><circle cx="10" cy="9" r="2.4"/><circle cx="20" cy="16" r="2.4"/><circle cx="14" cy="23" r="2.4"/></svg></div>
<h3>Multiple pricing engines</h3><p>Fourier (Lewis, COS, Carr–Madan), closed forms, Monte Carlo and Vibrato Monte Carlo, through one <code>pricer</code> interface.</p><p class="fc-why">Change the engine with one argument; speed and generality become a choice, not a rewrite.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><rect x="7" y="7" width="18" height="18" rx="3"/><rect x="12" y="12" width="8" height="8" rx="1"/><path d="M11 3v4M16 3v4M21 3v4M11 25v4M16 25v4M21 25v4M3 11h4M3 16h4M3 21h4M25 11h4M25 16h4M25 21h4"/></svg></div>
<h3>GPU acceleration</h3><p>Strikes and quotes as <code>CuArray</code>s: pricing, implied volatilities and derivatives stay on the device, in <code>Float64</code> or <code>Float32</code>.</p><p class="fc-why">6× the CPU in <code>Float32</code> on a 5000-strike smile, on a laptop GPU; a small surface is launch-bound.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><ellipse cx="16" cy="16" rx="12" ry="6"/><ellipse cx="16" cy="16" rx="7" ry="3.2"/><circle cx="16" cy="16" r="1.6" class="hl"/></svg></div>
<h3>Implicit differentiation</h3><p>At the optimum ∇L = 0, so ∂x*/∂v = H⁻¹Jᵀ: the derivative needs the conditions, not the path of the optimizer.</p><p class="fc-why">Matches −H⁻¹∂ᵥ∇L to 1e-14 in the tests; about 8 ms on top of the solve.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M4 26 16 6l12 20z"/><path d="M10 26 16 16l6 10"/></svg></div>
<h3>Multiple AD backends</h3><p>ForwardDiff, Zygote, Mooncake and Enzyme outside; any ADTypes backend inside, mixed freely. None is a hard dependency.</p><p class="fc-why">Use the backend your pipeline already uses; forward for few quotes, reverse for many.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M16 4v8M16 20v8M4 16h8M20 16h8"/><circle cx="16" cy="16" r="4"/></svg></div>
<h3>Extensible model framework</h3><p>Any FinancialMonteCarlo process: Black–Scholes, Merton, Kou, VG, NIG, Heston, lognormal mixtures. A new model–method pair is one <code>pricer</code> method.</p><p class="fc-why"><code>Parametrization</code> handles constraints and fixed parameters, so no step yields an invalid model.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M5 17l7 7L27 8"/></svg></div>
<h3>Production-ready workflows</h3><p>Out-of-the-money quotes filtered by vega, a Levenberg–Marquardt polish to a stationary point (|∇L| ~ 1e-8), Inf for unpriceable parameters, CBOE data loader.</p><p class="fc-why">Tested against recalibration and against a hand-written reference calibration.</p></article>
</div>
```

## How it fits together

```@raw html
<p class="fc-section-lede">Quotes flow left to right into risk; derivatives flow back along the same path. Each box is one function
call, and each arrow is differentiable.</p>
<div class="fc-arch">
<ol class="fc-flow">
<li><span class="fc-node-k">01</span><b>Market quotes</b><span>CBOE delayed quotes, JSON snapshots<br><code>retriever</code></span></li>
<li><span class="fc-node-k">02</span><b>Volatility surface</b><span>OTM vols, rates, implied dividends<br><code>adapter</code> → <code>MarketData</code></span></li>
<li class="fc-core"><span class="fc-node-k">03</span><b>Financial<wbr>Calibrator.jl</b><span>Optim + Levenberg–Marquardt; implicit function theorem<br><code>calibrate</code>, <code>calibrated_<wbr>parameters</code></span></li>
<li><span class="fc-node-k">04</span><b>Calibrated parameters</b><span>p*(v), with ∂p*/∂v<br><code>Parametrization</code></span></li>
<li><span class="fc-node-k">05</span><b>Pricing engines</b><span>FinancialFFT, closed form, Monte Carlo<br><code>pricer</code></span></li>
<li class="fc-out"><span class="fc-node-k">06</span><b>Risk sensitivities</b><span>∂V/∂vᵢ for every quote<br>one gradient</span></li>
</ol>
<div class="fc-backflow"><span>∂V/∂v</span> = <span>∇ₚV(p*)ᵀ</span> · <span>∂p*/∂v</span> <em>← chain rule through the pricer, then the implicit function theorem through the calibration</em></div>
</div>
```

## Results on a real surface

```@raw html
<p class="fc-section-lede">Computed by the package when this site is built, from the DAX surface that ships with it:
9 maturities from two months to four years, 12 strikes from 0.8 to 1.75 times the spot.</p>
<div class="fc-tabs">
<input type="radio" name="fc-results" id="fc-results-1" checked><label for="fc-results-1">Smiles</label>
<input type="radio" name="fc-results" id="fc-results-2"><label for="fc-results-2">Errors</label>
<input type="radio" name="fc-results" id="fc-results-3"><label for="fc-results-3">Parameter sensitivities</label>
<input type="radio" name="fc-results" id="fc-results-4"><label for="fc-results-4">Loss landscape</label>
<div class="fc-tab-panels">
<figure class="fc-tab-panel"><img src="assets/figures/home-smiles-dark.svg" alt="Model smiles and quotes for the nine maturities"><img src="assets/figures/home-smiles-light.svg" alt="Model smiles and quotes for the nine maturities"><figcaption><b>The smile at every maturity.</b> Closest around the money; the far wings are where one Heston model gives way.</figcaption></figure>
<figure class="fc-tab-panel"><img src="assets/figures/home-errors-dark.svg" alt="Heatmap of model minus market volatility in basis points"><img src="assets/figures/home-errors-light.svg" alt="Heatmap of model minus market volatility in basis points"><figcaption><b>Where the error is.</b> Model minus quoted volatility, in bp; the 9 quotes with too little vega are blank.</figcaption></figure>
<figure class="fc-tab-panel"><img src="assets/figures/home-sensitivities-dark.svg" alt="Heatmaps of the sensitivity of each Heston parameter to each quote"><img src="assets/figures/home-sensitivities-light.svg" alt="Heatmaps of the sensitivity of each Heston parameter to each quote"><figcaption><b>Which quotes move which parameter.</b> ∂p*/∂v from one ForwardDiff call: ρ follows the right wing, σ₀ the short end; κ is the weakest direction.</figcaption></figure>
<figure class="fc-tab-panel"><img src="assets/figures/home-landscape-dark.svg" alt="Contour lines of the loss over kappa and sigma"><img src="assets/figures/home-landscape-light.svg" alt="Contour lines of the loss over kappa and sigma"><figcaption><b>The landscape the optimizer sees.</b> The long valley in (κ, σ) is why κ is hard to pin down, and why the polish to a stationary point matters.</figcaption></figure>
</div>
</div>
```

## Benchmarks

```@raw html
<p class="fc-section-lede">Measured on a laptop CPU and GPU, medians after compilation: which choice to make, not a
headline number. Scaling, AD backends and pricing engines in detail: <a href="backends/">Methods, backends and
arrays</a>.</p>
```

```@example home
Main.DocumentationFigures.benchmark_cards()  # hide
```

```@raw html
<div class="fc-figure"><div class="fc-figure-head"><h3>Optimizers</h3><p>Every optimizer ends at the same optimum: the
Levenberg–Marquardt polish takes over from wherever it stops. <b>Insight:</b> pass <code>LBFGS()</code> or
<code>BFGS()</code> for Heston, Merton and NIG, 2 to 2.6× faster than the default Nelder–Mead.</p></div>
<img src="assets/figures/bench-optimizers-dark.svg" alt="Calibration time per model and optimizer"><img src="assets/figures/bench-optimizers-light.svg" alt="Calibration time per model and optimizer"></div>
```

## Use cases

```@raw html
<p class="fc-section-lede"><span class="fc-tag">built in</span> works out of the box;
<span class="fc-tag fc-tag-alt">compose</span> is built on top of it with the rest of the Julia ecosystem.</p>
<div class="fc-grid fc-cases">
<article class="fc-case"><header><h3>Equity volatility</h3><span class="fc-tag">built in</span></header><p>Fit Heston or a Lévy model to an index surface from DAX or CBOE data.</p><p class="fc-outcome">A stationary optimum and its error in bp, in about a second.</p></article>
<article class="fc-case"><header><h3>Exotic-option pricing</h3><span class="fc-tag">built in</span></header><p>Price American, Asian, barrier payoffs consistently with the vanillas.</p><p class="fc-outcome">A price and its vega to every quote, through simulation and calibration.</p></article>
<article class="fc-case"><header><h3>Risk management</h3><span class="fc-tag">built in</span></header><p>Bucketed vega of a book on a model that would need a recalibration per bump.</p><p class="fc-outcome">The whole sensitivity surface for about one calibration.</p></article>
<article class="fc-case"><header><h3>Model validation</h3><span class="fc-tag">built in</span></header><p>Are the parameters identified by the quotes? Is the optimum a true one?</p><p class="fc-outcome">Weak directions from the singular values of J, |∇L| at the optimum.</p></article>
<article class="fc-case"><header><h3>Differentiable research</h3><span class="fc-tag">built in</span></header><p>A calibration inside an objective: a bilevel problem.</p><p class="fc-outcome">Exact gradients of the outer objective, no unrolled optimizer.</p></article>
<article class="fc-case"><header><h3>FX volatility</h3><span class="fc-tag fc-tag-alt">compose</span></header><p>Delta quotes converted to strikes; the foreign rate as the dividend yield.</p><p class="fc-outcome">The same calibration and sensitivities as for equity.</p></article>
<article class="fc-case"><header><h3>Commodity derivatives</h3><span class="fc-tag fc-tag-alt">compose</span></header><p>One smile per futures expiry; the cost of carry as the dividend curve.</p><p class="fc-outcome">Parameters and their sensitivity to each quoted vol.</p></article>
<article class="fc-case"><header><h3>ML-assisted calibration</h3><span class="fc-tag fc-tag-alt">compose</span></header><p><code>calibrated_parameters</code> as a differentiable layer in a training loop.</p><p class="fc-outcome">Gradients that respect the calibration exactly, CPU or GPU.</p></article>
</div>
```

## Quick start

```@raw html
<p class="fc-section-lede">Six steps, two minutes, a real surface. Every output below was produced by this code when
the site was built.</p>
<div class="fc-step"><span class="fc-step-n">1</span><div><h3>Install the package</h3><p>FinancialFFT.jl is in a registry of its own, added once.</p></div></div>
```

```julia
] registry add https://github.com/rcalxrc08/LocalRegistry.git
] add https://github.com/rcalxrc08/FinancialCalibrator.jl
] add FinancialMonteCarlo, FinancialFFT, ForwardDiff
```

```@raw html
<div class="fc-step"><span class="fc-step-n">2</span><div><h3>Load market data</h3><p>The bundled DAX surface,
as the underlying and the <code>MarketData</code> of the calibration.</p></div></div>
```

```@example quickstart
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT, ForwardDiff

und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
size(mkt.Volatility)   # maturities × strikes
```

```@raw html
<div class="fc-step"><span class="fc-step-n">3</span><div><h3>Build a calibration problem</h3><p>A model at its
starting point, and a pricing engine.</p></div></div>
```

```@example quickstart
model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)   # σ, θ, κ: the starting point
method = LewisMethod(100.0, 4001)
nothing # hide
```

```@raw html
<div class="fc-step"><span class="fc-step-n">4</span><div><h3>Run the calibration</h3><p>Nelder–Mead by default, then
a Levenberg–Marquardt polish to a stationary point.</p></div></div>
```

```@example quickstart
res = calibrate(model, method, mkt)
res.OptimalParam
```

```@raw html
<div class="fc-step"><span class="fc-step-n">5</span><div><h3>Compute sensitivities</h3><p>∂p*/∂v for all 108
quotes, by the implicit function theorem.</p></div></div>
```

```@example quickstart
∂p_∂v = ForwardDiff.jacobian(v -> calibrated_parameters(model, method, mkt, v), mkt.Volatility)
size(∂p_∂v)   # parameters × quotes
```

```@raw html
<div class="fc-step"><span class="fc-step-n">6</span><div><h3>Analyse the results</h3><p>The fit in bp, the
worst quote, and the response to a parallel shift.</p></div></div>
```

```@example quickstart
e = 1e4 .* (res.ModelVolatility .- mkt.Volatility)       # errors in bp, NaN where left out
worst = argmax(i -> isnan(e[i]) ? -Inf : abs(e[i]), CartesianIndices(e))
(rms_bp = round(sqrt(sum(abs2, filter(!isnan, e)) / count(!isnan, e)); digits = 1),
 worst_quote = (T = mkt.TimeToMaturity[worst[1]], moneyness = round(mkt.Strikes[worst[2]] / und.S0; digits = 2), error_bp = round(e[worst]; digits = 1)),
 parallel_shift = round.(vec(sum(∂p_∂v; dims = 2)); digits = 3))   # ∂p*/∂h for every quote moved by h
```

```@raw html
<div class="fc-next">
<a href="calibration/"><b>Calibration, step by step</b><span>The loss, the polish, identifiability and the implicit derivatives, on Heston.</span></a>
<a href="backends/"><b>Methods, backends and arrays</b><span>Pricing engines, AD backends inside and out, GPU and Float32.</span></a>
<a href="exotics/"><b>Sensitivities of exotics</b><span>Monte Carlo positions and their vega to every quote.</span></a>
<a href="api/"><b>API reference</b><span>Every exported function and type.</span></a>
</div>
```

## The ecosystem

[FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl) defines the models, the payoffs and the
`pricer` interface, [FinancialFFT.jl](https://github.com/rcalxrc08/FinancialFFT.jl) the Fourier pricing methods, and
[FinancialToolbox.jl](https://github.com/rcalxrc08/FinancialToolbox.jl) the implied volatility `blsimpv`.
FinancialCalibrator.jl adds the loss, the optimization and the derivatives of the calibration. Market data comes from
the bundled DAX surface or from the free delayed quotes of the CBOE, see [Market data](api.md#Market-data). More
projects at [rcalxrc08.github.io](https://rcalxrc08.github.io/).
