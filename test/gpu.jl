using Test
using CUDA
using Adapt
using FinancialCalibrator
using FinancialMonteCarlo
using FinancialFFT
using Optim
using ForwardDiff
using Zygote

const FC = FinancialCalibrator

@testset "GPU" begin
    CUDA.allowscalar(false)
    dax_und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
    gmkt = adapt(CuArray, mkt)
    @testset "MarketData on the device" begin
        @test gmkt.Strikes isa CuVector{Float64}
        @test gmkt.Volatility isa CuMatrix{Float64}
        @test gmkt.TimeToMaturity isa Vector{Float64}
        @test Array(gmkt.Strikes) == mkt.Strikes
        @test Array(gmkt.Volatility) == mkt.Volatility
    end

    und = FinancialMonteCarlo.Underlying(dax_und.S0, 0.01)
    idx = 3
    args(m) = (m.Strikes, mkt.ZeroRates[idx], mkt.TimeToMaturity[idx])

    nig = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
    p0 = FinancialMonteCarlo.get_parameters(nig)
    @testset "smile_prices / $name" for (name, method) in ("CarrMadan" => CarrMadanMethod(400.0, 14), "CarrMadanLewis" => CarrMadanLewisMethod(400.0, 14), "Lewis" => LewisMethod(100.0, 4001), "Cos" => CosMethod(1024))
        cpu = smile_prices(nig, method, p0, args(mkt)...)
        gpu = smile_prices(nig, method, p0, args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-8)
        @test FC.loss(p0, nig, method, gmkt) ≈ FC.loss(p0, nig, method, mkt) rtol = 1e-6
        # a method in CudaMode with the strikes on the host: they go to the device
        @test isapprox(Array(smile_prices(nig, FinancialMonteCarlo.with_mode(method, FinancialMonteCarlo.CudaMode()), p0, args(mkt)...)), cpu; rtol = 1e-8)
    end

    @testset "smile_prices / closed form" begin
        slnm = ShiftedLogNormalMixture([0.2, 0.25, 0.3], [0.2, 0.3], -0.1, und)
        ps = FinancialMonteCarlo.get_parameters(slnm)
        cpu = smile_prices(slnm, AnalyticMethod(), ps, args(mkt)...)
        gpu = smile_prices(slnm, AnalyticMethod(), ps, args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-8)
        @test FC.loss(ps, slnm, AnalyticMethod(), gmkt) ≈ FC.loss(ps, slnm, AnalyticMethod(), mkt) rtol = 1e-6
        # in CudaMode with the strikes on the host: they go to the device
        on_device = smile_prices(slnm, AnalyticMethod(; mode = FinancialMonteCarlo.CudaMode()), ps, args(mkt)...)
        @test on_device isa CuVector
        @test isapprox(Array(on_device), cpu; rtol = 1e-8)
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

    # The calibrated parameters as a function of quotes on the device: the solve, the ForwardDiff derivative and the
    # Zygote gradient (a CuArray, like the quotes) against the host. Black-Scholes: ∂σ*/∂vᵢ is 1/n on the quotes in the loss.
    @testset "calibrated_parameters on the GPU" begin
        bs = BlackScholesProcess(0.3, dax_und)
        method = CosMethod(1024)
        @test calibrated_parameters(bs, method, gmkt) ≈ calibrated_parameters(bs, method, mkt) rtol = 1e-10
        i = CartesianIndex(4, 5)
        # no scalar indexing on the device
        bump(V, h) = (W = V .+ zero(h); W[i:i] .+= h; W)
        host = ForwardDiff.derivative(h -> calibrated_parameters(bs, method, mkt, bump(mkt.Volatility, h)), 0.0)
        device = ForwardDiff.derivative(h -> calibrated_parameters(bs, method, gmkt, bump(gmkt.Volatility, h)), 0.0)
        @test device ≈ host rtol = 1e-10
        g = Zygote.gradient(V -> sum(calibrated_parameters(bs, method, gmkt, V)), gmkt.Volatility)[1]
        @test g isa CuMatrix{Float64}
        @test Array(g) ≈ Zygote.gradient(V -> sum(calibrated_parameters(bs, method, mkt, V)), mkt.Volatility)[1] rtol = 1e-10
        @test Array(g)[i] ≈ host[1] rtol = 1e-10
    end
end
