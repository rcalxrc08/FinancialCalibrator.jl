using Test
using FinancialCalibrator
using Dates
using FinancialMonteCarlo
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
        # Neta = 2: eta (2), lambda (1), alpha (1)
        params = [0.2, 0.25, 0.5, -0.1]
        prices = FC.shiftedLognormalMixturePricer(S0, strikes, r, T, params, 0.0)
        @test length(prices) == 3
        @test all(prices .> 0)
        @test issorted(prices, rev = true)  # call prices decrease in strike
        # invalid parameters (negative vol) are flagged with a negative price
        bad = FC.shiftedLognormalMixturePricer(S0, strikes, r, T, [-0.2, 0.25, 0.5, -0.1], 0.0)
        @test all(bad .< 0)
    end

    @testset "Calibration" begin
        mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
        curve = FinancialMonteCarlo.ImpliedCurve(mkt.ImpliedDividend, mkt.TimeToMaturity)
        und = FinancialMonteCarlo.Underlying(mkt.S0, curve)

        @testset "Carr-Madan / NIG" begin
            cal = CalibratorCarrMadan(NormalInverseGaussianProcess(0.2, -0.03, 0.16, und))
            p0 = FinancialMonteCarlo.get_parameters(cal.Model)
            prices = FC.pricer(cal, mkt.S0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1], p0, mkt.ImpliedDividend[1])
            @test length(prices) == length(mkt.Strikes)
            @test all(prices .> 0)
            loss0 = FC.Loss(p0, cal, mkt)
            @test loss0 < 1e4
            res = calibrate(cal, mkt)
            @test res isa CalibratedData
            @test res.OptimalValue <= loss0
            @test size(res.ModelVolatility) == size(mkt.Volatility)
        end

        @testset "Shifted lognormal mixture" begin
            Random.seed!(0)
            cal = CalibratorShiftedLogNormalMixture(0.2 * ones(4), rand(3) / 4, -0.22, und)
            p0 = FinancialMonteCarlo.get_parameters(cal.Model)
            @test length(p0) == 8
            @test FC.Loss(p0, cal, mkt) < 1e4
        end
    end

    @testset "Abstract pricer is not implemented" begin
        struct DummyCalibrator <: FC.CalibratorBase end
        @test_throws ErrorException FC.pricer(DummyCalibrator(), 100.0, [100.0], 0.01, 1.0, [0.1], 0.0)
    end
end
