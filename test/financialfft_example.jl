# The calibration of the FinancialFFT documentation (https://rcalxrc08.github.io/FinancialFFT.jl/calibration/): Heston on
# four smiles of out of the money options, Levenberg–Marquardt on the volatility errors, and the sensitivities of the
# calibrated parameters to the quoted prices by the implicit function theorem, assembled by hand. The page's code runs
# as it is, and calibrate / calibrated_parameters must give the same parameters and the same sensitivities.
using Test
using FinancialCalibrator
using FinancialMonteCarlo
using FinancialFFT
using FinancialToolbox
using ForwardDiff
using LinearAlgebra
using Dates

@testset "The calibration of the FinancialFFT documentation" begin
    # ---- the page, verbatim ----
    S0, r, d = 100.0, 0.02, 0.01
    zero_rate = ZeroRate(r)
    maturities = [0.25, 0.5, 1.0, 2.0]
    strikes = collect(70.0:5.0:130.0)
    forward(T) = S0 * exp((r - d) * T)
    smiles = [FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, strikes .>= forward(T)) for T in maturities]
    quote_T = repeat(maturities; inner = length(strikes))
    quote_K = repeat(strikes, length(maturities))
    quote_call = quote_K .>= forward.(quote_T)

    method = CosMethod(2^9)
    heston(p) = HestonProcess(p[1], p[2], zero(p[1]), p[3], p[4], p[5], Underlying(S0, d))
    model_prices(p) = reduce(vcat, pricer.(heston(p), zero_rate, method, smiles))
    implied_vols(prices) = blsimpv.(S0, quote_K, r, quote_T, prices, d, quote_call)

    p_market = [0.5, 0.2, 1.5, -0.7, 0.04]
    market_prices = model_prices(p_market) .* (1 .+ 0.01 .* sin.(1:length(quote_K)))
    v = implied_vols(market_prices)

    to_p(x) = [exp(x[1]), exp(x[2]), exp(x[3]), tanh(x[4]), exp(x[5])]
    to_x(p) = [log(p[1]), log(p[2]), log(p[3]), atanh(p[4]), log(p[5])]
    vol_errors_page(x, v) = implied_vols(model_prices(to_p(x))) .- v
    function calibrate_page(v; x = to_x([0.3, 0.25, 1.0, -0.3, 0.06]), μ = 1e-2, iterations = 100)
        e = vol_errors_page(x, v)
        for _ = 1:iterations
            J = ForwardDiff.jacobian(t -> vol_errors_page(t, v), x)
            A = Array(transpose(J) * J)
            δ = -(A + μ * Diagonal(diag(A))) \ Array(transpose(J) * e)
            e_new = vol_errors_page(x + δ, v)
            if sum(abs2, e_new) < sum(abs2, e)
                x, e, μ = x + δ, e_new, μ / 3
                norm(δ) < 1e-12 && break
            else
                μ *= 4
                μ > 1e12 && break
            end
        end
        return x
    end
    x = calibrate_page(v)
    p = to_p(x)
    J = ForwardDiff.jacobian(t -> vol_errors_page(t, v), x)
    H = ForwardDiff.hessian(t -> sum(abs2, vol_errors_page(t, v)) / 2, x)
    vegas = blsvega.(S0, quote_K, r, quote_T, v, d)
    ∂p_∂m = ForwardDiff.jacobian(to_p, x) * ((H \ transpose(J)) ./ transpose(vegas))

    # ---- the same with FinancialCalibrator ----
    nT, nK = length(maturities), length(strikes)
    # the quotes are a matrix, one row per maturity: the page's vector runs maturity by maturity
    as_surface(w) = permutedims(reshape(w, nK, nT))
    mkt = MarketData(maturities, [Date(2026) + Day(round(Int, 365 * T)) for T in maturities], fill(r, nT), strikes, as_surface(v))
    # HestonProcess(σ, σ₀, λ, κ, ρ, θ): the page's variables, with λ = 0
    variables = Parametrization(y -> [exp(y[1]), exp(y[2]), zero(y[1]), exp(y[3]), tanh(y[4]), exp(y[5])], q -> [log(q[1]), log(q[2]), log(q[4]), atanh(q[5]), log(q[6])])
    start = HestonProcess(0.3, 0.25, 0.0, 1.0, -0.3, 0.06, Underlying(S0, d))
    # every quote, as on the page
    res = calibrate(start, method, mkt; parametrization = variables, min_vega = 0)
    model_params(q::AbstractVector) = q[[1, 2, 4, 5, 6]]
    model_params(q::AbstractMatrix) = q[[1, 2, 4, 5, 6], :]
    @test res.OptimalParam[3] == 0
    @test model_params(res.OptimalParam) ≈ p rtol = 1e-8
    # the root mean square error of the page, about 6 basis points of volatility
    @test sqrt(2 * res.OptimalValue / length(v)) ≈ sqrt(sum(abs2, vol_errors_page(x, v)) / length(v)) rtol = 1e-6
    @test res.ModelVolatility ≈ as_surface(implied_vols(model_prices(p))) rtol = 1e-8

    # ∂p*/∂v by ForwardDiff through calibrated_parameters, nothing assembled by hand; the quoted prices move the
    # quoted volatilities by 1/vega
    calibrated(w) = calibrated_parameters(start, method, mkt, as_surface(w); parametrization = variables, min_vega = 0)
    ∂p_∂v = ForwardDiff.jacobian(calibrated, v)
    @test all(iszero, ∂p_∂v[3, :])
    @test model_params(∂p_∂v) ./ transpose(vegas) ≈ ∂p_∂m rtol = 1e-6

    # The calibrated model prices what the quotes do not contain, here an Asian option by Monte Carlo (FinancialMonteCarlo,
    # the same seed at every pricing), and the sensitivity of its price to every quoted volatility is one gradient through
    # the pricing and the calibration. It is the chain rule (∂p*/∂v)ᵀ ∇ₚV, and recalibrating and repricing agrees.
    @testset "Sensitivity of an exotic to the quotes" begin
        mc = MonteCarloConfiguration(20_000, 50)
        asian = AsianFixedStrikeOption(1.0, 100.0)
        price(q) = pricer(FinancialMonteCarlo.set_parameters!(start, q), zero_rate, mc, asian)
        value(w) = price(calibrated(w))
        ∂V_∂v = ForwardDiff.gradient(value, v)
        @test ∂V_∂v ≈ transpose(∂p_∂v) * ForwardDiff.gradient(price, res.OptimalParam) rtol = 1e-8
        # small bumps: at the calibrated point the Feller condition does not hold, the variance of the paths reaches its
        # floor, and the price has kinks that a larger bump crosses
        i, h = 30, 1e-6
        bump(s) = (w = copy(v); w[i] += s; w)
        @test ∂V_∂v[i] ≈ (value(bump(h)) - value(bump(-h))) / 2h rtol = 1e-3
    end
end
