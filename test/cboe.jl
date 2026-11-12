# Market data of the CBOE delayed quotes, on a synthetic option chain in the format of the CBOE: no network and no
# market data in the tests (a real chain is megabytes; test/local/cboe_spx.jl uses one when it is on disk).
using Test
using Dates
using JSON
using FinancialCalibrator
using FinancialToolbox

const FC = FinancialCalibrator

# A chain of the SPX root on 12 November 2026, from Black and Scholes with a skew: the implied volatility of strike K at
# maturity T is skew_vol(K, T), every price is quoted with a bid and an ask 1% around it, so the mid is the price and
# put–call parity holds exactly.
const S0, RATE, DIVIDEND = 5000.0, 0.04, 0.01
skew_vol(K, T) = 0.2 - 0.15 * log(K / S0) / sqrt(1 + T)
function synthetic_chain(; expiries = Date.(["2026-11-20", "2026-12-18", "2027-03-19", "2027-06-17", "2027-12-17", "2028-12-15"]), strikes = 3000.0:100.0:7500.0)
    options = Dict{String, Any}[]
    for expiry in expiries, K in strikes, isCall in (true, false)
        T = Dates.value(expiry - Date(2026, 11, 12)) / 365
        σ = skew_vol(K, T)
        price = blsprice(S0, K, RATE, T, σ, DIVIDEND, isCall)
        symbol = "SPX" * Dates.format(expiry, "yymmdd") * (isCall ? "C" : "P") * lpad(round(Int, 1000 * K), 8, '0')
        push!(options, Dict("option" => symbol, "bid" => 0.99 * price, "ask" => 1.01 * price, "iv" => σ))
    end
    # a weekly option, a contract with no bid and one with a meaningless volatility: all left out
    push!(options, Dict("option" => "SPXW261124C05000000", "bid" => 50.0, "ask" => 51.0, "iv" => 0.2))
    push!(options, Dict("option" => "SPX271217C09000000", "bid" => 0.0, "ask" => 0.05, "iv" => 0.3))
    push!(options, Dict("option" => "SPX271217P01000000", "bid" => 0.05, "ask" => 0.1, "iv" => 6.3))
    return Dict("timestamp" => "2026-11-12 18:45:33", "data" => Dict("current_price" => S0, "options" => options))
end

@testset "CBOE delayed quotes" begin
    @test FC.parse_occ("SPX261218C05000000") == (root = "SPX", expiry = Date(2026, 12, 18), isCall = true, strike = 5000.0)
    @test FC.parse_occ("SPXW261124P07125500").strike == 7125.5
    @test_throws ArgumentError FC.parse_occ("SPX 261218C05000000")

    dir = mktempdir()
    write(FC.cboe_file("SPX", "12Nov2026", dir), JSON.json(synthetic_chain()))
    id = CalibrationID("SPX", "12Nov2026", "USD")
    raw = retriever(id, CboeInputStrategy(rate = RATE, dir = dir))
    @test raw isa RawMarketData
    @test raw.refDate == Date(2026, 11, 12)
    und, mkt = adapter(raw)
    @test und.S0 == S0
    # the adapter keeps the maturities between 0.1 and 5 years: four of the six expiries
    @test mkt.MaturityDates == Date.(["2027-03-19", "2027-06-17", "2027-12-17", "2028-12-15"])
    @test mkt.Strikes ≈ collect(0.8:0.05:1.2) .* S0
    @test all(mkt.ZeroRates .≈ RATE)
    # put–call parity on the mid prices recovers the forward, and the dividend yield
    @test all(FC.implied_dividend(und, t) ≈ DIVIDEND for t in mkt.TimeToMaturity)
    # the strikes of the grid are listed: the surface is the skew itself, out of the money puts below the forward and
    # calls above it giving the same volatility
    @test mkt.Volatility ≈ [skew_vol(K, T) for T in mkt.TimeToMaturity, K in mkt.Strikes] rtol = 1e-4
    # another grid; a root that is not in the chain
    @test size(adapter(retriever(id, CboeInputStrategy(moneyness = 0.9:0.1:1.1, dir = dir)))[2].Volatility) == (4, 3)
    @test_throws ArgumentError retriever(id, CboeInputStrategy(root = "SPXW", dir = dir))
    # a grid that no expiry covers
    @test_throws ArgumentError retriever(id, CboeInputStrategy(moneyness = 0.1:0.1:2.0, dir = dir))
    # the strategy that downloads the current chain (not called: no network in the tests)
    web = CboeWebInputStrategy(rate = 0.03, dir = "chains")
    @test web.rate == 0.03 && web.index && web.dir == "chains" && web.root === nothing
    @test CboeWebInputStrategy().dir === nothing
    @test FC.cboe_file("SPX", "12Nov2026", "chains") == joinpath("chains", "CboeOptions_SPX12Nov2026.json")
end

# The adapter selects the volatilities of the maturities it keeps (#2): a maturity below the lower threshold, first in
# the file, is dropped with its row.
@testset "Adapter with a maturity outside the thresholds" begin
    ref = "2020-01-01"
    spot = """[{"referenceDate":"$ref","value":100.0}]"""
    dates = ["2020-01-15", "2020-07-01", "2021-01-01", "2022-01-01", "2023-01-01"]
    curve = "[" * join(["""{"referenceDate":"$ref","maturityDate":"$d","value":0.01}""" for d in dates], ",") * "]"
    # the volatility of maturity i and moneyness m is i / 10 + m / 100
    vols = "[" * join(["""{"referenceDate":"$ref","maturityDate":"$d","pointValue":$m,"pointType":"MONEYNESS","value":$(i / 10 + m / 100)}""" for (i, d) in enumerate(dates) for m in (0.9, 1.0, 1.1)], ",") * "]"
    und, mkt = adapter(RawMarketData(Date(2020), spot, curve, curve, vols))
    @test mkt.MaturityDates == Date.(dates[2:end])
    @test mkt.Volatility ≈ [i / 10 + m / 100 for i = 2:5, m in (0.9, 1.0, 1.1)]
end
