<!-- =============================
     ABOUT
    ============================== -->

\begin{section}{title="About this Package", name="About"}

\lead{FinancialCalibrator.jl calibrates option pricing models to an implied volatility surface.}
Models and pricers come from [FinancialMonteCarlo.jl](https://github.com/rcalxrc08/FinancialMonteCarlo.jl) and [FinancialFFT.jl](https://github.com/rcalxrc08/FinancialFFT.jl), the minimization from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). It provides:
* **One entry point**: `calibrate(model, pricing_method, mkt)`. The pricing method decides how every smile is priced, numerically or in closed form.
* **Any model with a Fourier pricer**: Black and Scholes, Merton, Kou, Variance Gamma, Normal Inverse Gaussian, Heston.
* **Four numerical methods**: `CarrMadanMethod`, `CarrMadanLewisMethod`, `LewisMethod`, `CosMethod`.
* **A closed form** for the shifted lognormal mixture, through `AnalyticMethod()`.
* **Float32 and GPU**: the whole pipeline is generic in the floating point type, and strikes and volatilities can live on the GPU.
\end{section}


<!-- ==============================
     GETTING STARTED
     ============================== -->
\begin{section}{title="Getting started"}

FinancialFFT is published in the [LocalRegistry](https://github.com/rcalxrc08/LocalRegistry), add it first, then add the package:

```
pkg> registry add https://github.com/rcalxrc08/LocalRegistry.git
pkg> add https://github.com/rcalxrc08/FinancialCalibrator.jl
```
and load it:
```julia:ex__1
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT
```
\end{section}


<!-- ==============================
     USAGE
     ============================== -->
\begin{section}{title="Example of Usage", name="Usage"}

**Market data**:\
`adapter` reads the quotes and returns the underlying (spot and implied dividend curve) next to the market data (maturities, zero rates, strikes and volatilities). The sample data is the DAX surface of the 22nd of August 2016.
```julia:ex_data
id = CalibrationID("Dax", "22Aug2016", "EUR")
und, mkt = adapter(retriever(id, IOFileInputStrategy()))
println("Spot: ", und.S0)
println("Maturities: ", length(mkt.TimeToMaturity), ", strikes: ", length(mkt.Strikes))
```
\output{ex_data}

**Numerical calibration**:\
Spot and dividends are those of the model's underlying, so the model is built on the underlying returned by `adapter`. The second argument selects the pricing method.
```julia
model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
res = calibrate(model, CarrMadanMethod(400.0, 14), mkt)

res.OptimalParam      # calibrated parameters
res.OptimalValue      # value of the loss at the optimum
res.ModelVolatility   # model volatility surface, same shape as mkt.Volatility
```

**Closed form calibration**:\
```julia
model = ShiftedLogNormalMixture([0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1], -0.22, und)
res = calibrate(model, AnalyticMethod(), mkt)
```

**Optimizer settings**:\
Any Optim optimizer and `Optim.Options` can be given, nothing is printed unless `verbose = true`.
```julia
using Optim
calibrate(model, CosMethod(1024), mkt, NelderMead(), Optim.Options(iterations = 200); verbose = true)
```
\end{section}


<!-- ==============================
     MODELS AND METHODS
     ============================== -->
\begin{section}{title="Models and Methods", name="Methods"}

Every model can be priced with every numerical method, except where noted:

| Model | `CarrMadanMethod` | `CarrMadanLewisMethod` | `LewisMethod` | `CosMethod` | `AnalyticMethod` |
|:--|:-:|:-:|:-:|:-:|:-:|
| `BlackScholesProcess` | ✓ | ✓ | ✓ | ✓ | |
| `MertonProcess` | ✓ | ✓ | ✓ | ✓ | |
| `KouProcess` | ✓ | ✓ | ✓ | ✓ | |
| `VarianceGammaProcess` | ✓ | ✓ | ✓ | ✓ | |
| `NormalInverseGaussianProcess` | ✓ | ✓ | ✓ | ✓ | |
| `HestonProcess` | ✓ | ✓ | | | |
| `ShiftedLogNormalMixture` | | | | | ✓ |

Heston with `LewisMethod` and `CosMethod` returns negative prices on far out of the money strikes at the default parameters, so the loss falls back to its penalty. Adding a model and method pair only requires a `pricer` method from the FinancialMonteCarlo interface.
\end{section}


<!-- ==============================
     FLOAT32 AND GPU
     ============================== -->
\begin{section}{title="Float32 and GPU", name="GPU"}

The floating point type is chosen when reading the data, and the model follows:
```julia
und, mkt = adapter(retriever(id, IOFileInputStrategy()), Float32)
model = ShiftedLogNormalMixture(0.2f0 * ones(Float32, 4), fill(0.1f0, 3), -0.22f0, und)
calibrate(model, AnalyticMethod(), mkt)
```

With [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) loaded, strikes and volatilities can be moved to the device, and the smiles are priced there with every method (the execution mode of the method, part of the method since FinancialMonteCarlo 0.6, is moved to `CudaMode`):
```julia
using CUDA, Adapt
calibrate(model, CosMethod(1024), adapt(CuArray, mkt))
```
`CarrMadanMethod` does not run on the GPU yet, and `CarrMadanLewisMethod` needs a constant dividend underlying there.
\end{section}
