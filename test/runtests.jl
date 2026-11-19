using Test
using FinancialCalibrator
using Dates
using FinancialMonteCarlo
using FinancialFFT
using Random
using FinancialToolbox
using ForwardDiff
using Zygote
using LinearAlgebra

const FC = FinancialCalibrator

# Prices of the call smile of `model` re-parametrized with `p`, through the `pricer` interface.
# a method without a pricer: the loss must not hide the MethodError
struct NoPricer <: FinancialMonteCarlo.AbstractMethod end

function smile_prices(model, method, p, K, r, T)
    smile = FinancialMonteCarlo.EuropeanOptionSmile(T, K, fill!(similar(K, Bool), true))
    return vec(pricer(FinancialMonteCarlo.set_parameters!(model, p), ZeroRate(r), FC.calibration_method(method, K), smile))
end

# The testsets report only at the end, so each part logs when it starts and how long it took: a hung file shows up by
# name in the log (a local run or CI) instead of a silent process.
function logged(f, name)
    @info "Running $name"
    flush(stderr)
    elapsed = @elapsed f()
    @info "Finished $name in $(round(elapsed, digits = 1)) s"
    flush(stderr)
end
logged_include(file) = logged(() -> include(file), file)

@info "Running the core testset"
flush(stderr)
@testset "FinancialCalibrator" begin
    @testset "Data retrieval and adaptation" begin
        id = CalibrationID("Dax", "22Aug2016", "EUR")
        raw = retriever(id, IOFileInputStrategy())
        @test raw isa RawMarketData
        @test raw.refDate == Date(2016, 8, 22)

        und, mkt = adapter(raw)
        @test und isa FinancialMonteCarlo.AbstractUnderlying
        @test mkt isa MarketData
        @test und.S0 > 0
        nT, nK = size(mkt.Volatility)
        @test length(mkt.TimeToMaturity) == nT
        @test length(mkt.Strikes) == nK
        @test length(mkt.ZeroRates) == nT
        @test all(isfinite(FC.implied_dividend(und, t)) for t in mkt.TimeToMaturity)
        @test issorted(mkt.TimeToMaturity)
        @test all(0.1 .< mkt.TimeToMaturity .< 5.0)
        @test all(mkt.Volatility .> 0)
    end

    @testset "Shifted lognormal mixture pricer" begin
        S0, r, T = 100.0, 0.01, 1.0
        strikes = [80.0, 100.0, 120.0]
        und = FinancialMonteCarlo.Underlying(S0, 0.03)
        model = ShiftedProcess(LogNormalMixture((0.2, 0.25), (0.5,), und), -0.1)
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
        # and with the mixture of Black and Scholes prices on the shifted spot and strike
        shift = -0.1 * S0 * exp((r - 0.03) * T)
        mixture(K, isCall) = 0.5 * blsprice(1.1 * S0, K - shift, r, T, 0.2, 0.03, isCall) + 0.5 * blsprice(1.1 * S0, K - shift, r, T, 0.25, 0.03, isCall)
        @test pc ≈ mixture.(strikes, true) rtol = 1e-12
        @test pp ≈ mixture.(strikes, false) rtol = 1e-12
        # a positive shift and a strike below it: the call is a forward, the put is worthless
        up = ShiftedProcess(LogNormalMixture((0.2, 0.25), (0.5,), und), 0.5)
        low = FinancialMonteCarlo.EuropeanOptionSmile(T, [10.0, 10.0], [true, false])
        @test pricer(up, zr, AnalyticMethod(), low) ≈ [S0 * exp(-0.03 * T) - 10.0 * exp(-r * T), 0.0]
        # a shift that leaves no mixture
        @test_throws DomainError pricer(ShiftedProcess(LogNormalMixture((0.2, 0.25), (0.5,), und), 1.0), zr, AnalyticMethod(), calls)
        # the number of components is in the type of the model (tuples): the prices are inferred
        @test (@inferred pricer(model, zr, AnalyticMethod(), calls)) == pc
        @test (@inferred pricer(model, zr, AnalyticMethod(), FinancialMonteCarlo.EuropeanOption(T, 100.0))) ≈ pc[2]
        # no shift: the mixture itself; one component: the displaced diffusion
        mixture_model = LogNormalMixture((0.2, 0.25), (0.5,), und)
        @test pricer(mixture_model, zr, AnalyticMethod(), calls) ≈ pricer(ShiftedProcess(mixture_model, 0.0), zr, AnalyticMethod(), calls) rtol = 1e-14
        displaced = ShiftedProcess(BlackScholesProcess(0.2, und), -0.1)
        @test pricer(displaced, zr, AnalyticMethod(), calls) ≈ pricer(ShiftedProcess(LogNormalMixture((0.2, 0.2), (0.5,), und), -0.1), zr, AnalyticMethod(), calls) rtol = 1e-12
        # the Monte Carlo of FinancialMonteCarlo simulates the Brigo–Mercurio local volatility, whose European prices are
        # these closed forms: a calibrated mixture prices exotics by Monte Carlo consistently (#30)
        sobol = MonteCarloConfiguration(2^14, 50; sampling = FinancialMonteCarlo.SobolMC())
        @test pricer(model, zr, sobol, calls) ≈ pc atol = 0.05
        # the kernel is straight code for Zygote (#39): no branch, no tuple `map`, no `oftype`, so its pullback is inferred
        # (an uninferred one made the gradient 33 times slower); the gradient against ForwardDiff, also where κ ≤ 0
        for (m, K) in ((model, 100.0), (up, 10.0)), isCall in (true, false)
            k = FC.analytic_kernel(m, S0, r, 0.03, T)
            price_of(k) = k(K, isCall)
            @test !occursin("Any", string(only(Base.return_types(Zygote.pullback, Tuple{typeof(price_of), typeof(k)}))))
            g = Zygote.gradient(price_of, k)[1]
            fd = ForwardDiff.gradient(x -> FC.LognormalMixtureKernel(x[1], x[2], (x[3], x[4]), (x[5], x[6]), r, T)(K, isCall), [k.F, k.shift, k.σ..., k.w...])
            @test [g.F, g.shift, g.σ..., g.w...] ≈ fd rtol = 1e-12
        end
    end

    @testset "Black and Scholes, closed form" begin
        S0, r, d, T, σ = 100.0, 0.01, 0.03, 1.0, 0.2
        und = FinancialMonteCarlo.Underlying(S0, d)
        bs = BlackScholesProcess(σ, und)
        strikes = collect(40.0:10.0:250.0)
        for isCall in (true, false)
            smile = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, fill(isCall, length(strikes)))
            @test (@inferred pricer(bs, ZeroRate(r), AnalyticMethod(), smile)) ≈ blsprice.(S0, strikes, r, T, σ, d, isCall) rtol = 1e-12
        end
        # a shifted lognormal mixture with no shift and equal volatilities is Black and Scholes
        smile = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, strikes .>= S0)
        @test pricer(ShiftedProcess(LogNormalMixture((σ, σ), (0.3,), und), 0.0), ZeroRate(r), AnalyticMethod(), smile) ≈ pricer(bs, ZeroRate(r), AnalyticMethod(), smile) rtol = 1e-12
        # far out of the money, where 1 + erf cancels: the implied volatility is still recovered
        far = FinancialMonteCarlo.EuropeanOptionSmile(T, [20.0, 400.0], [false, true])
        prices = pricer(bs, ZeroRate(r), AnalyticMethod(), far)
        @test all(prices .> 0)
        @test blsimpv.(S0, [20.0, 400.0], r, T, prices, d, [false, true]) ≈ [σ, σ] rtol = 1e-6
        # calibration: the volatility of a Black and Scholes surface
        mkt = MarketData([0.5, 1.0, 2.0], Date(2020) .+ Day.([182, 365, 730]), fill(r, 3), strikes[4:10], fill(0.25, 3, 7))
        @test calibrate(bs, AnalyticMethod(), mkt).OptimalParam ≈ [0.25] rtol = 1e-8
    end

    @testset "Calibration" begin
        und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))

        @testset "Carr-Madan / NIG" begin
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
            method = CarrMadanMethod(400.0, 14)
            p0 = FinancialMonteCarlo.get_parameters(model)
            prices = smile_prices(model, method, p0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
            @test length(prices) == length(mkt.Strikes)
            @test all(prices .> 0)
            loss0 = FC.loss(p0, model, method, mkt)
            @test loss0 < 1e4
            res = calibrate(model, method, mkt)
            @test res isa CalibratedData
            @test res.OptimalValue <= loss0
            @test size(res.ModelVolatility) == size(mkt.Volatility)
        end

        # Every AbstractMethod can calibrate (FinancialMonteCarlo 0.6): a Monte Carlo configuration, with the same seed at
        # every evaluation of the loss. On the strikes near the money and the longer maturities: far out of the money a few
        # thousand paths price zero, which the loss penalises.
        @testset "Monte Carlo / Black Scholes" begin
            m, k = 4:length(mkt.TimeToMaturity), 4:9
            near = MarketData(mkt.TimeToMaturity[m], mkt.MaturityDates[m], mkt.ZeroRates[m], mkt.Strikes[k], mkt.Volatility[m, k])
            model = BlackScholesProcess(0.25, und)
            mc = MonteCarloConfiguration(2_000, 1; sampling = FinancialMonteCarlo.AntitheticMC())
            loss0 = FC.loss(FinancialMonteCarlo.get_parameters(model), model, mc, near)
            @test loss0 < 1e4
            @test FC.loss(FinancialMonteCarlo.get_parameters(model), model, mc, near) == loss0
            res = calibrate(model, mc, near, FC.Optim.NelderMead(), FC.Optim.Options(iterations = 10))
            @test res.OptimalValue <= loss0
        end

        # The loss and the closed-form pricer are inferred: set_parameters! is not, hence the function barrier in loss.
        @testset "Inferred" begin
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
            p0 = FinancialMonteCarlo.get_parameters(model)
            @test (@inferred FC.loss(p0, model, CosMethod(256), mkt)) isa Float64
            # also with dual numbers, in the parameters and in the quotes (the derivatives of the implicit function)
            dual = ForwardDiff.Dual{Nothing}.(p0, 1.0)
            @test (@inferred FC.vol_errors(dual, model, CosMethod(256), mkt)) isa Vector{eltype(dual)}
            @test (@inferred FC.vol_errors(p0, model, CosMethod(256), mkt; vols = ForwardDiff.Dual{Nothing}.(mkt.Volatility, 1.0))) isa Vector{eltype(dual)}
            slnm = ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3), und), -0.1)
            smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, mkt.Strikes, fill(true, length(mkt.Strikes)))
            @test (@inferred pricer(slnm, ZeroRate(0.01), AnalyticMethod(), smile)) isa Vector{Float64}
            @test (@inferred pricer(slnm, ZeroRate(0.01), AnalyticMethod(), FinancialMonteCarlo.EuropeanOption(1.0, 100.0))) isa Float64
            @test (@inferred FC.loss(FinancialMonteCarlo.get_parameters(slnm), slnm, AnalyticMethod(), mkt)) isa Float64
            # the solver of calibrated_parameters (Optim, then the polish, whose Jacobian is not inferred): Enzyme's rules
            # for the implicit function need it inferred
            bs = BlackScholesProcess(0.2, und)
            cache = FC.loss_cache(bs, CosMethod(256), mkt)
            options = FC.Optim.Options(iterations = 5)
            @test (@inferred FC.optimal_variables(bs, CosMethod(256), mkt, cache, FC.DEFAULT_OPTIMIZER, options, Parametrization())) isa Vector{Float64}
        end

        # The volatility errors: one per quote, maturity by maturity, out of the money options only, generic in the quotes.
        @testset "vol_errors" begin
            model = BlackScholesProcess(0.2, und)
            method = CosMethod(1024)
            nT, nK = size(mkt.Volatility)
            cache = FC.loss_cache(model, method, mkt)
            # the quotes of tiny vega are left out (9 on this surface, at the short maturities and high strikes), and none
            # with min_vega = 0
            @test sum(length, cache.quoted) == nT * nK - 9
            @test sum(length, FC.loss_cache(model, method, mkt; min_vega = 0).quoted) == nT * nK
            # out of the money: puts below the forward, calls above
            @test all(!first(c) && last(c) for c in cache.is_call)
            e = vol_errors([0.2], model, method, mkt, cache)
            quoted = reduce(vcat, [mkt.Volatility[i, q] for (i, q) in enumerate(cache.quoted)])
            @test length(e) == length(quoted)
            # Black-Scholes prices invert to its own volatility, puts and calls alike
            @test e ≈ 0.2 .- quoted atol = 1e-12
            @test FC.loss([0.2], model, method, mkt) ≈ sum(abs2, e) / 2
            # derivatives with respect to the quotes: minus the selection of the quotes in the loss
            J = ForwardDiff.jacobian(v -> vol_errors([0.2], model, method, mkt, cache; vols = reshape(v, nT, nK)), vec(mkt.Volatility))
            @test count(==(-1), J) == length(e) && count(iszero, J) == length(J) - length(e)
            @test all(==(-1), sum(J; dims = 2))
            @test count(iszero, sum(J; dims = 1)) == 9
            # a price that is not positive is a DomainError: the loss is Inf there
            @test_throws DomainError vol_errors([0.2], model, MonteCarloConfiguration(10, 1), mkt)
            @test FC.loss([0.2], model, MonteCarloConfiguration(10, 1), mkt) == Inf
            # parameters the model rejects (a DomainError of its constructor) too; any other exception propagates (#3)
            @test FC.loss([-0.2], model, method, mkt) == Inf
            @test_throws MethodError FC.loss([0.2], model, NoPricer(), mkt)
            # Zygote differentiates the volatility errors and the loss (its try included) of a closed form: the prices
            # are one allocating broadcast, not a write into an array (#39)
            slnm = ShiftedProcess(LogNormalMixture((0.1, 0.2, 0.35), (0.3, 0.4), und), -0.1)
            p = [0.1, 0.2, 0.35, 0.3, 0.4, -0.1]
            scache = FC.loss_cache(slnm, AnalyticMethod(), mkt)
            half_squares(x) = sum(abs2, vol_errors(x, slnm, AnalyticMethod(), mkt, scache)) / 2
            reference = ForwardDiff.gradient(half_squares, p)
            @test Zygote.gradient(half_squares, p)[1] ≈ reference rtol = 1e-12
            @test Zygote.gradient(x -> FC.loss(x, slnm, AnalyticMethod(), mkt, scache), p)[1] ≈ reference rtol = 1e-12
        end

        # The Levenberg–Marquardt polish ends where the gradient of the loss vanishes, on the market surface too.
        @testset "Stationary on the market" begin
            model = BlackScholesProcess(0.3, und)
            method = CosMethod(1024)
            res = calibrate(model, method, mkt)
            g = ForwardDiff.gradient(p -> FC.loss(p, model, method, mkt), res.OptimalParam)
            @test norm(g) < 1e-9
            # without it, NelderMead stops at its own tolerances
            @test calibrate(model, method, mkt; polish = false).OptimalValue >= res.OptimalValue
            # no model volatility at the quotes left out of the loss
            @test count(isnan, res.ModelVolatility) == 9
            # the model volatilities come from the errors of the polish: the same as pricing the surface again
            cache = FC.loss_cache(model, method, mkt)
            calibrated = FinancialMonteCarlo.set_parameters!(model, res.OptimalParam)
            @test all(res.ModelVolatility[i, cache.quoted[i]] ≈ FC.model_volatilities(calibrated, und.S0, mkt, cache, i) for i in eachindex(cache.quoted) if !isempty(cache.quoted[i]))
            @test res.OptimalValue ≈ FC.loss(res.OptimalParam, model, method, mkt)
        end

        # The default parametrizations: the inverse of each other on valid parameters, and every variable a valid model
        # (the constraints between the parameters of VG, NIG and the mixtures included), so the loss is finite (#24).
        @testset "Default parametrizations" begin
            models = (BlackScholesProcess(0.2, und), MertonProcess(0.15, 0.3, -0.1, 0.2, und), KouProcess(0.15, 0.5, 0.4, 10.0, 15.0, und), VarianceGammaProcess(0.2, -0.03, 0.16, und), NormalInverseGaussianProcess(0.2, -0.03, 0.16, und), HestonProcess(0.3, 0.2, 0.0, 1.5, -0.6, 0.04, und), ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3), und), -0.1), LogNormalMixture((0.2, 0.3), (0.4,), und))
            for model in models
                P = default_parametrization(model)
                p = FinancialMonteCarlo.get_parameters(model)
                x = P.from_params(p)
                @test P.to_params(x) ≈ p
                for y in (x .+ 3, x .- 3, -x, zero(x))
                    @test FinancialMonteCarlo.set_parameters!(model, P.to_params(y)) isa typeof(model)
                end
            end
            # Heston: λ is not calibrated, it is zero
            @test default_parametrization(models[6]).to_params(zeros(5))[3] == 0
        end

        # The calibrated parameters as a function of the quotes, through the implicit function theorem. On Black-Scholes
        # every quote in the loss weighs the same: ∂σ*/∂vᵢ = 1/n.
        @testset "calibrated_parameters" begin
            model = BlackScholesProcess(0.3, und)
            method = CosMethod(1024)
            n = sum(length, FC.loss_cache(model, method, mkt).quoted)
            @test calibrated_parameters(model, method, mkt) ≈ calibrate(model, method, mkt).OptimalParam
            i = CartesianIndex(4, 5)
            bump(h) = (V = mkt.Volatility .+ zero(h); V[i] += h; V)
            ∂σ = ForwardDiff.derivative(h -> calibrated_parameters(model, method, mkt, bump(h)), 0.0)
            @test ∂σ ≈ [1 / n] rtol = 1e-8
            h = 1e-4
            @test ∂σ ≈ (calibrated_parameters(model, method, mkt, bump(h)) - calibrated_parameters(model, method, mkt, bump(-h))) / 2h rtol = 1e-4
            @test_throws DimensionMismatch calibrated_parameters(model, method, mkt, mkt.Volatility[:, 2:end])
        end

        @testset "Numerical methods agree" begin
            p0 = [0.2, -0.03, 0.16]
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
            args = (mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
            ref = smile_prices(model, CarrMadanMethod(400.0, 14), p0, args...)
            for m in (CarrMadanLewisMethod(400.0, 14), CosMethod(256))
                @test isapprox(smile_prices(model, m, p0, args...), ref; rtol = 1e-3)
            end
        end

        @testset "Shifted lognormal mixture" begin
            Random.seed!(0)
            model = ShiftedProcess(LogNormalMixture(ntuple(_ -> 0.2, 4), Tuple(rand(3) / 4), und), -0.22)
            method = AnalyticMethod()
            p0 = FinancialMonteCarlo.get_parameters(model)
            @test length(p0) == 8
            @test FC.loss(p0, model, method, mkt) < 1e4
        end
    end

    @testset "Float32" begin
        raw = retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy())
        und, mkt = adapter(raw, Float32)
        @test und.S0 isa Float32
        @test mkt isa MarketData{Float32}
        @test eltype(mkt.Volatility) == Float32

        model = ShiftedProcess(LogNormalMixture(ntuple(_ -> 0.2f0, 4), ntuple(_ -> 0.1f0, 3), und), -0.22f0)
        p0 = FinancialMonteCarlo.get_parameters(model)
        @test eltype(p0) == Float32
        prices = smile_prices(model, AnalyticMethod(), p0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
        @test eltype(prices) == Float32
        @test FC.loss(p0, model, AnalyticMethod(), mkt) isa Float32

        nig = NormalInverseGaussianProcess(0.2f0, -0.03f0, 0.16f0, und)
        method = CarrMadanMethod(400.0f0, 14, 0.75f0)
        @test FC.loss(FinancialMonteCarlo.get_parameters(nig), nig, method, mkt) isa Float32
    end

    @testset "AnalyticMethod on a model without closed form" begin
        und = FinancialMonteCarlo.Underlying(100.0)
        model = NormalInverseGaussianProcess(0.2, -0.03, 0.16, und)
        @test_throws MethodError smile_prices(model, AnalyticMethod(), [0.2, -0.03, 0.16], [100.0], 0.01, 1.0)
    end
end
@info "Finished the core testset"
flush(stderr)

logged_include("models_methods.jl")
logged_include("financialfft_example.jl")
logged_include("cboe.jl")
# market data that is not in the repository: run where it is on disk
if isfile(joinpath(@__DIR__, "..", "data", "json", "cboe", "2026-11-12", "CboeOptions_SPX12Nov2026.json"))
    logged_include(joinpath("local", "cboe_spx.jl"))
else
    @info "No SPX option chain in data/json/cboe/2026-11-12: skipping the local tests"
end
logged_include("aqua.jl")

using CUDA
if CUDA.functional()
    logged_include("gpu.jl")
else
    @info "No functional GPU: skipping the GPU tests"
end
