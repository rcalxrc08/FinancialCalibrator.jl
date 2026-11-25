using Test
using CUDA
using Adapt
using FinancialCalibrator
using FinancialMonteCarlo
using FinancialFFT
using Optim
using ForwardDiff
using Zygote
using Mooncake
using Enzyme
using ADTypes: AutoMooncake, AutoEnzyme
import DifferentiationInterface as DI

const FC = FinancialCalibrator

@testset "GPU" begin
    progress("GPU")
    CUDA.allowscalar(false)
    dax_und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
    gmkt = adapt(CuArray, mkt)
    @testset "MarketData on the device" begin
        progress("MarketData on the device")
        @test gmkt.Strikes isa CuVector{Float64}
        @test gmkt.Volatility isa CuMatrix{Float64}
        @test gmkt.TimeToMaturity isa Vector{Float64}
        @test Array(gmkt.Strikes) == mkt.Strikes
        @test Array(gmkt.Volatility) == mkt.Volatility
    end

    und = FinancialMonteCarlo.Underlying(dax_und.S0, 0.01)
    idx = 3
    args(m) = (m.Strikes, mkt.ZeroRates[idx], mkt.TimeToMaturity[idx])

    nig = NormalInverseGaussianProcess(0.2, -0.03, 0.16)
    p0 = FinancialMonteCarlo.get_parameters(nig)
    @testset "smile_prices / $name" for (name, method) in ("CarrMadan" => CarrMadanMethod(400.0, 14), "CarrMadanLewis" => CarrMadanLewisMethod(400.0, 14), "Lewis" => LewisMethod(100.0, 4001), "Cos" => CosMethod(1024))
        progress("smile_prices / $name")
        cpu = smile_prices(nig, und, method, p0, args(mkt)...)
        gpu = smile_prices(nig, und, method, p0, args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-8)
        @test FC.loss(p0, nig, und, method, gmkt) ≈ FC.loss(p0, nig, und, method, mkt) rtol = 1e-6
        # a method in CudaMode with the strikes on the host: they go to the device
        @test isapprox(Array(smile_prices(nig, und, FinancialMonteCarlo.with_mode(method, FinancialMonteCarlo.CudaMode()), p0, args(mkt)...)), cpu; rtol = 1e-8)
    end

    @testset "smile_prices / closed form" begin
        progress("smile_prices / closed form")
        slnm = ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3)), -0.1)
        ps = FinancialMonteCarlo.get_parameters(slnm)
        cpu = smile_prices(slnm, und, AnalyticMethod(), ps, args(mkt)...)
        gpu = smile_prices(slnm, und, AnalyticMethod(), ps, args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-8)
        @test FC.loss(ps, slnm, und, AnalyticMethod(), gmkt) ≈ FC.loss(ps, slnm, und, AnalyticMethod(), mkt) rtol = 1e-6
        # in CudaMode with the strikes on the host: they go to the device
        on_device = smile_prices(slnm, und, AnalyticMethod(; mode = FinancialMonteCarlo.CudaMode()), ps, args(mkt)...)
        @test on_device isa CuVector
        @test isapprox(Array(on_device), cpu; rtol = 1e-8)
    end

    # Zygote on the device differentiates the fields of the closed-form kernel by dual numbers, which its fields typed by
    # type parameters can hold (it returned nothing, #41)
    @testset "Zygote gradient of the closed form on the GPU" begin
        progress("Zygote gradient of the closed form on the GPU")
        slnm = ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3)), -0.1)
        ps = FinancialMonteCarlo.get_parameters(slnm)
        host = ForwardDiff.gradient(q -> FC.loss(q, slnm, und, AnalyticMethod(), mkt), ps)
        device = Zygote.gradient(q -> FC.loss(q, slnm, und, AnalyticMethod(), gmkt), ps)[1]
        @test device ≈ host rtol = 1e-8
        J = Zygote.jacobian(q -> FC.vol_errors(q, slnm, und, AnalyticMethod(), gmkt), ps)[1]
        @test Array(J) ≈ ForwardDiff.jacobian(q -> FC.vol_errors(q, slnm, und, AnalyticMethod(), mkt), ps) rtol = 1e-8
    end

    # Mooncake on the device: the prices of the maturities are concatenated by reduce(vcat), whose GPU kernel Mooncake
    # cannot differentiate (a bitcast); the extension gives it a rule (#41). The maturities keep different numbers of quotes.
    @testset "Mooncake gradient on the GPU" begin
        progress("Mooncake gradient on the GPU")
        mooncake = AutoMooncake(; config = nothing)
        x = CuArray([1.0, 2.0, 3.0])
        g = DI.gradient(x -> sum(abs2, reduce(vcat, [x .* 2, x[1:2] .* 3])), mooncake, x)
        @test Array(g) ≈ [26.0, 52.0, 24.0]
        method = LewisMethod(100.0, 4001)
        host = ForwardDiff.gradient(q -> FC.loss(q, nig, und, method, mkt), p0)
        @test DI.gradient(q -> FC.loss(q, nig, und, method, gmkt), mooncake, p0) ≈ host rtol = 1e-8
    end

    # The closed form broadcasts a callable with state, which Mooncake's device broadcast refuses: the extension's rule
    # differentiates the same prices with the numbers of the kernel as broadcast arguments (#41 group 3c). A mixture of
    # three components (12 broadcast arguments) with the dividend curve of the DAX, and Black–Scholes.
    @testset "Mooncake gradient of the closed form on the GPU / $name" for (name, model) in ("slnm" => ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3)), -0.1), "bs" => BlackScholesProcess(0.2))
        progress("Mooncake gradient of the closed form on the GPU / $name")
        ps = FinancialMonteCarlo.get_parameters(model)
        host = ForwardDiff.gradient(q -> FC.loss(q, model, dax_und, AnalyticMethod(), mkt), ps)
        @test DI.gradient(q -> FC.loss(q, model, dax_und, AnalyticMethod(), gmkt), AutoMooncake(; config = nothing), ps) ≈ host rtol = 1e-10
    end

    @testset "smile_prices / Black and Scholes closed form" begin
        progress("smile_prices / Black and Scholes closed form")
        bs = BlackScholesProcess(0.2)
        cpu = smile_prices(bs, und, AnalyticMethod(), [0.2], args(mkt)...)
        gpu = smile_prices(bs, und, AnalyticMethod(), [0.2], args(gmkt)...)
        @test gpu isa CuVector
        @test isapprox(Array(gpu), cpu; rtol = 1e-10)
    end

    @testset "calibrate on the GPU" begin
        progress("calibrate on the GPU")
        method = CosMethod(1024)
        opts = Optim.Options(iterations = 5)
        cpu = calibrate(nig, und, method, mkt, NelderMead(), opts)
        gpu = calibrate(nig, und, method, gmkt, NelderMead(), opts)
        @test gpu.ModelVolatility isa Matrix{Float64}
        @test isapprox(gpu.OptimalParam, cpu.OptimalParam; rtol = 1e-4)
        @test gpu.OptimalValue ≈ cpu.OptimalValue rtol = 1e-4
    end

    # The calibrated parameters as a function of quotes on the device: the solve, the ForwardDiff derivative and the
    # Zygote gradient (a CuArray, like the quotes) against the host. Black-Scholes: ∂σ*/∂vᵢ is 1/n on the quotes in the loss.
    @testset "calibrated_parameters on the GPU" begin
        progress("calibrated_parameters on the GPU")
        bs = BlackScholesProcess(0.3)
        method = CosMethod(1024)
        @test calibrated_parameters(bs, dax_und, method, gmkt) ≈ calibrated_parameters(bs, dax_und, method, mkt) rtol = 1e-10
        i = CartesianIndex(4, 5)
        # no scalar indexing on the device
        bump(V, h) = (W = V .+ zero(h); W[i:i] .+= h; W)
        host = ForwardDiff.derivative(h -> calibrated_parameters(bs, dax_und, method, mkt, bump(mkt.Volatility, h)), 0.0)
        device = ForwardDiff.derivative(h -> calibrated_parameters(bs, dax_und, method, gmkt, bump(gmkt.Volatility, h)), 0.0)
        @test device ≈ host rtol = 1e-10
        g = Zygote.gradient(V -> sum(calibrated_parameters(bs, dax_und, method, gmkt, V)), gmkt.Volatility)[1]
        @test g isa CuMatrix{Float64}
        @test Array(g) ≈ Zygote.gradient(V -> sum(calibrated_parameters(bs, dax_und, method, mkt, V)), mkt.Volatility)[1] rtol = 1e-10
        @test Array(g)[i] ≈ host[1] rtol = 1e-10
        # Mooncake from outside: the setup of the implicit function (copies from the device) has a zero derivative
        gm = DI.gradient(V -> sum(calibrated_parameters(bs, dax_und, method, gmkt, V)), AutoMooncake(; config = nothing), gmkt.Volatility)
        @test Array(gm) ≈ Array(g) rtol = 1e-10
        # Enzyme from outside: its rules for calibrated_variables build the implicit function outside of the
        # differentiated code (#21)
        ge = DI.gradient(V -> sum(calibrated_parameters(bs, dax_und, method, gmkt, V)), AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const), gmkt.Volatility)
        @test ge isa CuMatrix{Float64}
        @test Array(ge) ≈ Array(g) rtol = 1e-10
    end
end
