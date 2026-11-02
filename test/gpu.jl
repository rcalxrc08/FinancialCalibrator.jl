using Test
using CUDA
using Adapt
using FinancialCalibrator
using FinancialMonteCarlo
using FinancialFFT
using Optim

const FC = FinancialCalibrator

@testset "GPU" begin
    CUDA.allowscalar(false)
    mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
    gmkt = adapt(CuArray, mkt)
    @testset "MarketData on the device" begin
        @test gmkt.Strikes isa CuVector{Float64}
        @test gmkt.Volatility isa CuMatrix{Float64}
        @test gmkt.TimeToMaturity isa Vector{Float64}
        @test Array(gmkt.Strikes) == mkt.Strikes
        @test Array(gmkt.Volatility) == mkt.Volatility
    end

    und = FinancialMonteCarlo.Underlying(mkt.S0, 0.01)
    idx = 3
    args(m) = (m.Strikes, mkt.ZeroRates[idx], mkt.TimeToMaturity[idx])

    nig = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
    p0 = FinancialMonteCarlo.get_parameters(nig)
    @testset "smile_prices / $name" for (name, method) in ("CarrMadanLewis" => CarrMadanLewisMethod(400.0, 14), "Lewis" => LewisMethod(100.0, 4001), "Cos" => CosMethod(1024))
        cpu = FC.smile_prices(nig, method, p0, args(mkt)...)
        gpu = FC.smile_prices(nig, method, p0, args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-8)
        @test FC.Loss(p0, nig, method, gmkt) ≈ FC.Loss(p0, nig, method, mkt) rtol = 1e-6
    end

    @testset "smile_prices / closed form" begin
        slnm = ShiftedLogNormalMixture([0.2, 0.25, 0.3], [0.2, 0.3], -0.1, und)
        ps = FinancialMonteCarlo.get_parameters(slnm)
        cpu = FC.smile_prices(slnm, AnalyticMethod(), ps, args(mkt)...)
        gpu = FC.smile_prices(slnm, AnalyticMethod(), ps, args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-8)
        @test FC.Loss(ps, slnm, AnalyticMethod(), gmkt) ≈ FC.Loss(ps, slnm, AnalyticMethod(), mkt) rtol = 1e-6
    end

    @testset "calibrate on the GPU" begin
        method = CosMethod(1024)
        opts = Optim.Options(iterations = 5)
        cpu = calibrate(nig, method, mkt, NelderMead(), opts)
        gpu = calibrate(nig, method, gmkt, NelderMead(), opts)
        @test gpu.ModelVolatility isa Matrix{Float64}
        @test isapprox(gpu.OptimalParam, cpu.OptimalParam; rtol = 1e-4)
        @test gpu.OptimalValue ≈ cpu.OptimalValue rtol = 1e-4
    end
end
