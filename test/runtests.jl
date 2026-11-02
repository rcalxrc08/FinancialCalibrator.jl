using Test
using FinancialCalibrator
using Dates
using FinancialMonteCarlo
using FinancialFFT
using Random

const FC = FinancialCalibrator

@testset "FinancialCalibrator" begin
    @testset "Data retrieval and adaptation" begin
        id = CalibrationID("Dax", "22Aug2016", "EUR")
        raw = retriever(id, IOFileInputStrategy())
        @test raw isa RawMarketData
        @test raw.refDate == Date(2016, 8, 22)

        mkt = adapter(raw)
        @test mkt isa MarketData
        @test mkt.S0 > 0
        nT, nK = size(mkt.Volatility)
        @test length(mkt.TimeToMaturity) == nT
        @test length(mkt.Strikes) == nK
        @test length(mkt.ZeroRates) == nT
        @test length(mkt.ImpliedDividend) == nT
        @test issorted(mkt.TimeToMaturity)
        @test all(0.1 .< mkt.TimeToMaturity .< 5.0)
        @test all(mkt.Volatility .> 0)
    end

    @testset "Shifted lognormal mixture pricer" begin
        S0, r, T = 100.0, 0.01, 1.0
        strikes = [80.0, 100.0, 120.0]
        und = FinancialMonteCarlo.Underlying(S0, 0.03)
        model = ShiftedLogNormalMixture([0.2, 0.25], [0.5], -0.1, und)
        zr = ZeroRate(r)
        calls = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, fill(true, 3))
        puts = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, fill(false, 3))
        pc = pricer(model, zr, AnalyticMethod(), calls)
        pp = pricer(model, zr, AnalyticMethod(), puts)
        @test length(pc) == 3
        @test all(pc .> 0)
        @test issorted(pc, rev = true)  # call prices decrease in strike
        # put-call parity
        @test pc .- pp ≈ S0 * exp(-0.03 * T) .- strikes .* exp(-r * T)
        # a single option agrees with the smile
        @test pricer(model, zr, AnalyticMethod(), FinancialMonteCarlo.EuropeanOption(T, 100.0)) ≈ pc[2]
        # and with the scalar formula
        @test pc ≈ FC.shiftedLognormalMixture.(S0, strikes, r, T, Ref([0.2, 0.25]), Ref([0.5]), -0.1, 0.03)
        # invalid parameters are flagged with a negative price
        @test FC.shiftedLognormalMixture(S0, 100.0, r, T, [-0.2, 0.25], [0.5], -0.1, 0.0) < 0
    end

    @testset "Calibration" begin
        mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
        curve = FinancialMonteCarlo.ImpliedCurve(mkt.ImpliedDividend, mkt.TimeToMaturity)
        und = FinancialMonteCarlo.Underlying(mkt.S0, curve)

        @testset "Carr-Madan / NIG" begin
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
            method = CarrMadanMethod(400.0, 14)
            p0 = FinancialMonteCarlo.get_parameters(model)
            prices = FC.smile_prices(model, method, p0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
            @test length(prices) == length(mkt.Strikes)
            @test all(prices .> 0)
            loss0 = FC.Loss(p0, model, method, mkt)
            @test loss0 < 1e4
            res = calibrate(model, method, mkt)
            @test res isa CalibratedData
            @test res.OptimalValue <= loss0
            @test size(res.ModelVolatility) == size(mkt.Volatility)
        end

        @testset "Numerical methods agree" begin
            p0 = [0.2, -0.03, 0.16]
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
            args = (mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
            ref = FC.smile_prices(model, CarrMadanMethod(400.0, 14), p0, args...)
            for m in (CarrMadanLewisMethod(400.0, 14), CosMethod(256))
                @test isapprox(FC.smile_prices(model, m, p0, args...), ref; rtol = 1e-3)
            end
        end

        @testset "Shifted lognormal mixture" begin
            Random.seed!(0)
            model = ShiftedLogNormalMixture(0.2 * ones(4), rand(3) / 4, -0.22, und)
            method = AnalyticMethod()
            p0 = FinancialMonteCarlo.get_parameters(model)
            @test length(p0) == 8
            @test FC.Loss(p0, model, method, mkt) < 1e4
        end
    end

    @testset "Float32" begin
        raw = retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy())
        mkt = adapter(raw, Float32)
        @test mkt isa MarketData{Float32}
        @test eltype(mkt.Volatility) == Float32
        curve = FinancialMonteCarlo.ImpliedCurve(mkt.ImpliedDividend, mkt.TimeToMaturity)
        und = FinancialMonteCarlo.Underlying(mkt.S0, curve)

        model = ShiftedLogNormalMixture(0.2f0 * ones(Float32, 4), fill(0.1f0, 3), -0.22f0, und)
        p0 = FinancialMonteCarlo.get_parameters(model)
        @test eltype(p0) == Float32
        prices = FC.smile_prices(model, AnalyticMethod(), p0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
        @test eltype(prices) == Float32
        @test FC.Loss(p0, model, AnalyticMethod(), mkt) isa Float32

        nig = NormalInverseGaussianProcess(0.2f0, -0.03f0, 0.16f0, und)
        method = CarrMadanMethod(400.0f0, 14, 0.75f0)
        @test FC.Loss(FinancialMonteCarlo.get_parameters(nig), nig, method, mkt) isa Float32
    end

    @testset "AnalyticMethod on a model without closed form" begin
        und = FinancialMonteCarlo.Underlying(100.0)
        model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
        @test_throws ErrorException FC.smile_prices(model, AnalyticMethod(), [0.2, -0.03, 0.16], [100.0], 0.01, 1.0)
    end
end

include("models_methods.jl")

using CUDA
if CUDA.functional()
    include("gpu.jl")
else
    @info "No functional GPU: skipping the GPU tests"
end
