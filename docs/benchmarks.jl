# The benchmarks of the home page, measured on the author's laptop and saved in src/assets/benchmarks.toml, which
# docs/figures.jl draws: timings are not taken on the documentation runner, whose hardware changes and has no GPU.
# Run in an environment with the package and the methods (CUDA is optional; without a GPU the device rows are skipped):
#
#   using Pkg; Pkg.activate(temp = true); Pkg.develop(path = "."); Pkg.add(["FinancialFFT", "FinancialMonteCarlo", "ForwardDiff", "Optim", "CUDA", "Adapt", "TOML"])
#   include("docs/benchmarks.jl")
#
# Every timing is the median of several runs after one run that compiles.
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT, ForwardDiff, Optim, Statistics, TOML, Dates

median_time(f; runs = 5) = (f(); median([@elapsed(f()) for _ = 1:runs]))
rmse_bp(res) = 1e4 * sqrt(2 * res.OptimalValue / count(!isnan, res.ModelVolatility))

und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
lewis = LewisMethod(100.0, 4001)

# 1. Optimizers: the first stage of calibrate, followed by the same Levenberg–Marquardt polish
models = ["Black–Scholes" => BlackScholesProcess(0.2, und), "Merton" => MertonProcess(0.2, 0.5, -0.1, 0.2, und), "Kou" => KouProcess(0.2, 0.5, 0.4, 10.0, 15.0, und), "Variance Gamma" => VarianceGammaProcess(0.2, -0.03, 0.16, und), "NIG" => NormalInverseGaussianProcess(0.2, -0.03, 0.16, und), "Heston" => HestonProcess(0.3, 0.2, 0.0, 1.0, -0.5, 0.04, und)]
optimizers = ["NelderMead" => NelderMead(), "LBFGS" => LBFGS(), "BFGS" => BFGS()]
optimizer_rows = Dict{String, Any}[]
for (name, model) in models, (opt_name, opt) in optimizers
    res = calibrate(model, lewis, mkt, opt)
    t = median_time(() -> calibrate(model, lewis, mkt, opt))
    push!(optimizer_rows, Dict("model" => name, "optimizer" => opt_name, "seconds" => t, "rmse_bp" => rmse_bp(res), "parameters" => res.OptimalParam))
    println(name, " ", opt_name, ": ", round(t; digits = 3), " s, ", round(rmse_bp(res); digits = 2), " bp")
end

# 2. Pricing engines: Heston, the same surface, three Fourier methods
engines = ["CosMethod(2^10)" => CosMethod(2^10), "LewisMethod(100, 4001)" => lewis, "CarrMadanLewisMethod(600, 16)" => CarrMadanLewisMethod(600.0, 16)]
engine_rows = Dict{String, Any}[]
heston = last(models[end])
for (name, method) in engines
    cache = FinancialCalibrator.loss_cache(heston, method, mkt)
    p = FinancialMonteCarlo.get_parameters(heston)
    t_loss = median_time(() -> FinancialCalibrator.loss(p, heston, method, mkt, cache); runs = 50)
    res = calibrate(heston, method, mkt)
    t = median_time(() -> calibrate(heston, method, mkt))
    push!(engine_rows, Dict("engine" => name, "loss_ms" => 1e3 * t_loss, "calibration_seconds" => t, "rmse_bp" => rmse_bp(res)))
    println(name, ": loss ", round(1e3 * t_loss; digits = 2), " ms, calibration ", round(t; digits = 3), " s")
end

# 3. Scaling: one loss on a single smile of n strikes (Black–Scholes, CosMethod(4096)), on the host and on the device
function smile_market(n, ::Type{T}) where {T}
    strikes = collect(range(T(60), T(160); length = n))
    vols = permutedims(T(0.2) .+ T(0.1) .* (strikes ./ 100 .- 1) .^ 2)
    return MarketData(T[1], [Date(2027)], T[0.01], strikes, vols)
end
sizes = [12, 50, 200, 500, 1000, 2000, 5000]
bs(T) = BlackScholesProcess(T(0.2), Underlying(T(100), T(0)))
host_ms(n, T) = (m = smile_market(n, T); c = FinancialCalibrator.loss_cache(bs(T), CosMethod(4096), m; min_vega = 0); 1e3 * median_time(() -> FinancialCalibrator.loss(T[0.21], bs(T), CosMethod(4096), m, c); runs = 20))
scaling = Dict{String, Any}("strikes" => sizes, "host_float64_ms" => [host_ms(n, Float64) for n in sizes])
gpu = try
    @eval using CUDA, Adapt
    Base.invokelatest(() -> CUDA.functional())
catch
    false
end
if gpu
    device_ms(n, T) = Base.invokelatest() do
        m = Adapt.adapt(CUDA.CuArray, smile_market(n, T))
        c = FinancialCalibrator.loss_cache(bs(T), CosMethod(4096), m; min_vega = 0)
        1e3 * median_time(() -> (FinancialCalibrator.loss(T[0.21], bs(T), CosMethod(4096), m, c); CUDA.synchronize()); runs = 20)
    end
    scaling["device_float64_ms"] = [device_ms(n, Float64) for n in sizes]
    scaling["device_float32_ms"] = [device_ms(n, Float32) for n in sizes]
    scaling["device"] = Base.invokelatest(() -> CUDA.name(CUDA.device()))
end
println(scaling)

benchmarks = Dict("measured" => string(today()), "cpu" => strip(Sys.cpu_info()[1].model), "threads" => Threads.nthreads(), "julia" => string(VERSION), "surface" => "DAX, 22 August 2016: 9 maturities × 12 strikes, 99 quotes in the loss", "optimizers" => optimizer_rows, "engines" => engine_rows, "scaling" => scaling)
open(joinpath(@__DIR__, "src", "assets", "benchmarks.toml"), "w") do io
    TOML.print(io, benchmarks; sorted = true)
end
