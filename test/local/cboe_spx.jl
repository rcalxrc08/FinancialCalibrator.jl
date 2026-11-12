# Run locally only: the real option chain of the SPX on 12 November 2026, as downloaded from the CBOE (13.8 MB, in the
# untracked and git-ignored data/json/real, see its README). runtests.jl includes this file only when the chain is there.
# Normal Inverse Gaussian is calibrated to the surface, and the calibrated parameters are differentiated with respect to
# the quotes.
using Test
using Dates
using FinancialCalibrator
using FinancialMonteCarlo
using FinancialFFT
using ForwardDiff
using LinearAlgebra

const SPX_DIR = joinpath(@__DIR__, "..", "..", "data", "json", "real")

@testset "SPX from the CBOE (local)" begin
    und, mkt = adapter(retriever(CalibrationID("SPX", "12Nov2026", "USD"), CboeInputStrategy(rate = 0.04, dir = SPX_DIR)))
    @test und.S0 ≈ 7625.79
    @test size(mkt.Volatility, 2) == 9
    @test size(mkt.Volatility, 1) >= 10
    @test all(0.05 .< mkt.Volatility .< 0.6)
    # the skew of an equity index
    @test all(mkt.Volatility[:, 1] .> mkt.Volatility[:, 5])

    # NIG, NormalInverseGaussianProcess(σ, θ, κ), with σ and κ positive; Lewis, whose derivatives are those of its
    # prices (not Cos for NIG, FinancialFFT#64)
    variables = Parametrization(x -> [exp(x[1]), x[2], exp(x[3])], p -> [log(p[1]), p[2], log(p[3])])
    start = NormalInverseGaussianProcess(0.15, -0.1, 0.5, und)
    method = LewisMethod(400.0, 40001)
    res = calibrate(start, method, mkt; parametrization = variables)
    rms = sqrt(2 * res.OptimalValue / count(!isnan, res.ModelVolatility))
    @info "NIG on the SPX surface" parameters = res.OptimalParam rms
    @test rms < 0.02
    # the calibration ends where the gradient of the loss vanishes
    cache = FinancialCalibrator.loss_cache(start, method, mkt)
    @test norm(ForwardDiff.gradient(x -> FinancialCalibrator.loss(variables.to_params(x), start, method, mkt, cache), variables.from_params(res.OptimalParam))) < 1e-8

    # the sensitivity of the calibrated parameters to one quote, by the implicit function theorem, against recalibrating
    calibrated(V) = calibrated_parameters(start, method, mkt, V; parametrization = variables)
    i = CartesianIndex(3, 5)
    bump(h) = (V = mkt.Volatility .+ zero(h); V[i] += h; V)
    ∂p = ForwardDiff.derivative(h -> calibrated(bump(h)), 0.0)
    # 1e-4: smaller bumps are below the stopping tolerance of the recalibrations (1e-6 is off by 2%)
    h = 1e-4
    @test ∂p ≈ (calibrated(bump(h)) - calibrated(bump(-h))) / 2h rtol = 1e-3
end
