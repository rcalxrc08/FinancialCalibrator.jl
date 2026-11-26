# Progress of a long run (~45 minutes): testsets report only when they end, so every file and testset logs its start
# with the time since the run began, and the log (a local run, Kaimon or CI) shows where a run is and what is slow.
# Loading the packages is timed too: on a fresh environment it precompiles for ~15 minutes before the first test.
const RUN_START = time()
function progress(name)
    @info "[$(round(Int, time() - RUN_START)) s] $name"
    flush(stderr)
end
progress("Loading the packages")

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
progress("Packages loaded")

# Prices of the call smile of `model` re-parametrized with `p`, through the `pricer` interface.
# a method without a pricer: the loss must not hide the MethodError
struct NoPricer <: FinancialMonteCarlo.AbstractMethod end
# a method whose derivatives fail where its values do not: the closed form, and a DomainError on dual numbers. The loss
# must not turn that into Inf and a zero gradient (#42).
struct DerivativeFails <: FinancialMonteCarlo.AbstractMethod end
failing_derivative(::ForwardDiff.Dual, _) = throw(DomainError(NaN, "a derivative that fails"))
failing_derivative(::Real, prices) = prices
FinancialMonteCarlo.pricer(model::BlackScholesProcess, und::FinancialMonteCarlo.AbstractUnderlying, r::FinancialMonteCarlo.AbstractZeroRateCurve, ::DerivativeFails, smile::FinancialMonteCarlo.EuropeanOptionSmile) = failing_derivative(model.σ, pricer(model, und, r, AnalyticMethod(), smile))

function smile_prices(model, und, method, p, K, r, T)
    smile = FinancialMonteCarlo.EuropeanOptionSmile(T, K, fill!(similar(K, Bool), true))
    return vec(pricer(FinancialMonteCarlo.set_parameters!(model, p), und, ZeroRate(r), FC.calibration_method(method, K), smile))
end

# Each file logs when it starts and how long it took: a hung file shows up by name instead of a silent process.
function logged(f, name)
    progress("Running $name")
    elapsed = @elapsed f()
    progress("Finished $name in $(round(elapsed, digits = 1)) s")
end
logged_include(file) = logged(() -> include(file), file)

progress("Running the core testset")
@testset "FinancialCalibrator" begin
    progress("FinancialCalibrator")
    @testset "Data retrieval and adaptation" begin
        progress("Data retrieval and adaptation")
        id = CalibrationID("Dax", "22Aug2016", "EUR")
        raw = retriever(id, IOFileInputStrategy())
        @test raw isa RawMarketData
        @test raw.refDate == Date(2016, 8, 22)
        # the same id by keywords or with a Date; swapped or malformed arguments fail at once (#6)
        @test CalibrationID(currency = "EUR", underlying = "Dax", refDate = "22Aug2016") == id
        @test CalibrationID("Dax", Date(2016, 8, 22), "EUR") == id
        @test_throws ArgumentError CalibrationID("22Aug2016", "Dax", "EUR")
        @test_throws ArgumentError CalibrationID("Dax", "2016-08-22", "EUR")
        @test_throws ArgumentError CalibrationID("EUR", "22Aug2016", "Dax")

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
        progress("Shifted lognormal mixture pricer")
        S0, r, T = 100.0, 0.01, 1.0
        strikes = [80.0, 100.0, 120.0]
        und = FinancialMonteCarlo.Underlying(S0, 0.03)
        model = ShiftedProcess(LogNormalMixture((0.2, 0.25), (0.5,)), -0.1)
        zr = ZeroRate(r)
        calls = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, fill(true, 3))
        puts = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, fill(false, 3))
        pc = pricer(model, und, zr, AnalyticMethod(), calls)
        pp = pricer(model, und, zr, AnalyticMethod(), puts)
        @test length(pc) == 3
        @test all(pc .> 0)
        @test issorted(pc, rev = true)  # call prices decrease in strike
        # put-call parity
        @test pc .- pp ≈ S0 * exp(-0.03 * T) .- strikes .* exp(-r * T)
        # a single option agrees with the smile
        @test pricer(model, und, zr, AnalyticMethod(), FinancialMonteCarlo.EuropeanOption(T, 100.0)) ≈ pc[2]
        # and with the mixture of Black and Scholes prices on the shifted spot and strike
        shift = -0.1 * S0 * exp((r - 0.03) * T)
        mixture(K, isCall) = 0.5 * blsprice(1.1 * S0, K - shift, r, T, 0.2, 0.03, isCall) + 0.5 * blsprice(1.1 * S0, K - shift, r, T, 0.25, 0.03, isCall)
        @test pc ≈ mixture.(strikes, true) rtol = 1e-12
        @test pp ≈ mixture.(strikes, false) rtol = 1e-12
        # a positive shift and a strike below it: the call is a forward, the put is worthless
        up = ShiftedProcess(LogNormalMixture((0.2, 0.25), (0.5,)), 0.5)
        low = FinancialMonteCarlo.EuropeanOptionSmile(T, [10.0, 10.0], [true, false])
        @test pricer(up, und, zr, AnalyticMethod(), low) ≈ [S0 * exp(-0.03 * T) - 10.0 * exp(-r * T), 0.0]
        # a shift that leaves no mixture
        @test_throws DomainError pricer(ShiftedProcess(LogNormalMixture((0.2, 0.25), (0.5,)), 1.0), und, zr, AnalyticMethod(), calls)
        # the number of components is in the type of the model (tuples): the prices are inferred
        @test (@inferred pricer(model, und, zr, AnalyticMethod(), calls)) == pc
        @test (@inferred pricer(model, und, zr, AnalyticMethod(), FinancialMonteCarlo.EuropeanOption(T, 100.0))) ≈ pc[2]
        # no shift: the mixture itself; one component: the displaced diffusion
        mixture_model = LogNormalMixture((0.2, 0.25), (0.5,))
        @test pricer(mixture_model, und, zr, AnalyticMethod(), calls) ≈ pricer(ShiftedProcess(mixture_model, 0.0), und, zr, AnalyticMethod(), calls) rtol = 1e-14
        displaced = ShiftedProcess(BlackScholesProcess(0.2), -0.1)
        @test pricer(displaced, und, zr, AnalyticMethod(), calls) ≈ pricer(ShiftedProcess(LogNormalMixture((0.2, 0.2), (0.5,)), -0.1), und, zr, AnalyticMethod(), calls) rtol = 1e-12
        # the Monte Carlo of FinancialMonteCarlo simulates the Brigo–Mercurio local volatility, whose European prices are
        # these closed forms: a calibrated mixture prices exotics by Monte Carlo consistently (#30)
        sobol = MonteCarloConfiguration(2^14, 50; sampling = FinancialMonteCarlo.SobolMC())
        @test pricer(model, und, zr, sobol, calls) ≈ pc atol = 0.05
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
        progress("Black and Scholes, closed form")
        S0, r, d, T, σ = 100.0, 0.01, 0.03, 1.0, 0.2
        und = FinancialMonteCarlo.Underlying(S0, d)
        bs = BlackScholesProcess(σ)
        strikes = collect(40.0:10.0:250.0)
        for isCall in (true, false)
            smile = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, fill(isCall, length(strikes)))
            @test (@inferred pricer(bs, und, ZeroRate(r), AnalyticMethod(), smile)) ≈ blsprice.(S0, strikes, r, T, σ, d, isCall) rtol = 1e-12
        end
        # a shifted lognormal mixture with no shift and equal volatilities is Black and Scholes
        smile = FinancialMonteCarlo.EuropeanOptionSmile(T, strikes, strikes .>= S0)
        @test pricer(ShiftedProcess(LogNormalMixture((σ, σ), (0.3,)), 0.0), und, ZeroRate(r), AnalyticMethod(), smile) ≈ pricer(bs, und, ZeroRate(r), AnalyticMethod(), smile) rtol = 1e-12
        # far out of the money, where 1 + erf cancels: the implied volatility is still recovered
        far = FinancialMonteCarlo.EuropeanOptionSmile(T, [20.0, 400.0], [false, true])
        prices = pricer(bs, und, ZeroRate(r), AnalyticMethod(), far)
        @test all(prices .> 0)
        @test blsimpv.(S0, [20.0, 400.0], r, T, prices, d, [false, true]) ≈ [σ, σ] rtol = 1e-6
        # calibration: the volatility of a Black and Scholes surface
        mkt = MarketData([0.5, 1.0, 2.0], Date(2020) .+ Day.([182, 365, 730]), fill(r, 3), strikes[4:10], fill(0.25, 3, 7))
        @test calibrate(bs, und, AnalyticMethod(), mkt).OptimalParam ≈ [0.25] rtol = 1e-8
    end

    @testset "Calibration" begin
        progress("Calibration")
        und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))

        @testset "Carr-Madan / NIG" begin
            progress("Carr-Madan / NIG")
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16)
            method = CarrMadanMethod(400.0, 14)
            p0 = FinancialMonteCarlo.get_parameters(model)
            prices = smile_prices(model, und, method, p0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
            @test length(prices) == length(mkt.Strikes)
            @test all(prices .> 0)
            loss0 = FC.loss(p0, model, und, method, mkt)
            @test loss0 < 1e4
            res = calibrate(model, und, method, mkt)
            @test res isa CalibratedData
            @test res.OptimalValue <= loss0
            @test size(res.ModelVolatility) == size(mkt.Volatility)
        end

        # Every AbstractMethod can calibrate (FinancialMonteCarlo 0.6): a Monte Carlo configuration, with the same seed at
        # every evaluation of the loss. On the strikes near the money and the longer maturities: far out of the money a few
        # thousand paths price zero, which the loss penalises.
        @testset "Monte Carlo / Black Scholes" begin
            progress("Monte Carlo / Black Scholes")
            m, k = 4:length(mkt.TimeToMaturity), 4:9
            near = MarketData(mkt.TimeToMaturity[m], mkt.MaturityDates[m], mkt.ZeroRates[m], mkt.Strikes[k], mkt.Volatility[m, k])
            model = BlackScholesProcess(0.25)
            mc = MonteCarloConfiguration(2_000, 1; sampling = FinancialMonteCarlo.AntitheticMC())
            loss0 = FC.loss(FinancialMonteCarlo.get_parameters(model), model, und, mc, near)
            @test loss0 < 1e4
            @test FC.loss(FinancialMonteCarlo.get_parameters(model), model, und, mc, near) == loss0
            res = calibrate(model, und, mc, near, FC.Optim.NelderMead(), FC.Optim.Options(iterations = 10))
            @test res.OptimalValue <= loss0
        end

        # calibrate says when the optimizer stops at its iteration or time limit (#46), and the result keeps its status
        @testset "Optimizer at its limit" begin
            progress("Optimizer at its limit")
            bs = BlackScholesProcess(0.2)
            res = @test_logs calibrate(bs, und, AnalyticMethod(), mkt)
            @test res.Converged && res.Iterations > 0 && res.LossCalls >= res.Iterations
            short = @test_logs (:warn, r"iteration limit") calibrate(bs, und, AnalyticMethod(), mkt, FC.Optim.NelderMead(), FC.Optim.Options(iterations = 2))
            @test !short.Converged && short.Iterations == 2
            @test short.OptimalParam ≈ res.OptimalParam rtol = 1e-8   # the polish still reaches the optimum
            # a line search that needs many losses per iteration gets the hint about the pricing method
            rosenbrock(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
            wrong = FC.Optim.optimize(rosenbrock, x -> [1.0, 1.0], [-1.2, 1.0], FC.Optim.LBFGS(), FC.Optim.Options(iterations = 3); inplace = false)
            @test_logs (:warn, r"iteration limit.*inaccurate pricing method") FC.warn_if_stopped_by_limit(wrong)
            slow = FC.Optim.optimize(x -> (sleep(0.02); rosenbrock(x)), [-1.2, 1.0], FC.Optim.NelderMead(), FC.Optim.Options(time_limit = 0.2))
            @test_logs (:warn, r"time limit") FC.warn_if_stopped_by_limit(slow)
        end

        # The loss and the closed-form pricer are inferred: set_parameters! is not, hence the function barrier in loss.
        @testset "Inferred" begin
            progress("Inferred")
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16)
            p0 = FinancialMonteCarlo.get_parameters(model)
            @test (@inferred FC.loss(p0, model, und, CosMethod(256), mkt)) isa Float64
            # also with dual numbers, in the parameters and in the quotes (the derivatives of the implicit function)
            dual = ForwardDiff.Dual{Nothing}.(p0, 1.0)
            @test (@inferred FC.vol_errors(dual, model, und, CosMethod(256), mkt)) isa Vector{eltype(dual)}
            @test (@inferred FC.vol_errors(p0, model, und, CosMethod(256), mkt; vols = ForwardDiff.Dual{Nothing}.(mkt.Volatility, 1.0))) isa Vector{eltype(dual)}
            slnm = ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3)), -0.1)
            smile = FinancialMonteCarlo.EuropeanOptionSmile(1.0, mkt.Strikes, fill(true, length(mkt.Strikes)))
            @test (@inferred pricer(slnm, und, ZeroRate(0.01), AnalyticMethod(), smile)) isa Vector{Float64}
            @test (@inferred pricer(slnm, und, ZeroRate(0.01), AnalyticMethod(), FinancialMonteCarlo.EuropeanOption(1.0, 100.0))) isa Float64
            @test (@inferred FC.loss(FinancialMonteCarlo.get_parameters(slnm), slnm, und, AnalyticMethod(), mkt)) isa Float64
            # the solver of calibrated_parameters (Optim, then the polish): Enzyme's rules for the implicit function need it
            # inferred
            bs = BlackScholesProcess(0.2)
            cache = FC.loss_cache(bs, und, CosMethod(256), mkt)
            options = FC.Optim.Options(iterations = 5)
            @test (@inferred FC.optimal_variables(bs, CosMethod(256), mkt, cache, FC.DEFAULT_OPTIMIZER, options, Parametrization())) isa Vector{Float64}
            # the polish (#36): the ForwardDiff chunk from the number of parameters of the model, so the prepared Jacobian,
            # and the loop computed from it, are inferred; a parametrization that changes the number of variables keeps
            # the backend as given
            x0 = FinancialMonteCarlo.get_parameters(slnm)
            # (a Union of the two backends, split at the call of the polish: the length of x is a run-time value)
            backend = FC.polish_backend_for(FC.default_backend(), slnm, x0)
            @test backend isa FC.ADTypes.AutoForwardDiff{6}
            @test FC.polish_backend_for(FC.default_backend(), slnm, x0[1:5]) === FC.default_backend()
            @test FC.polish_backend_for(FC.ADTypes.AutoFiniteDiff(), slnm, x0) === FC.ADTypes.AutoFiniteDiff()
            errors(x) = FC.vol_errors(x, slnm, und, AnalyticMethod(), mkt)
            @test isconcretetype(only(Base.return_types(FC.DI.prepare_jacobian, (typeof(errors), typeof(backend), typeof(x0)))))
        end

        # The volatility errors: one per quote, maturity by maturity, out of the money options only, generic in the quotes.
        @testset "vol_errors" begin
            progress("vol_errors")
            model = BlackScholesProcess(0.2)
            method = CosMethod(1024)
            nT, nK = size(mkt.Volatility)
            cache = FC.loss_cache(model, und, method, mkt)
            # the quotes of tiny vega are left out (9 on this surface, at the short maturities and high strikes), and none
            # with min_vega = 0
            @test sum(length, cache.quoted) == nT * nK - 9
            @test sum(length, FC.loss_cache(model, und, method, mkt; min_vega = 0).quoted) == nT * nK
            # out of the money: puts below the forward, calls above
            @test all(!first(c) && last(c) for c in cache.is_call)
            e = vol_errors([0.2], model, method, mkt, cache)
            quoted = reduce(vcat, [mkt.Volatility[i, q] for (i, q) in enumerate(cache.quoted)])
            @test length(e) == length(quoted)
            # Black-Scholes prices invert to its own volatility, puts and calls alike
            @test e ≈ 0.2 .- quoted atol = 1e-12
            @test FC.loss([0.2], model, und, method, mkt) ≈ sum(abs2, e) / 2
            # derivatives with respect to the quotes: minus the selection of the quotes in the loss
            J = ForwardDiff.jacobian(v -> vol_errors([0.2], model, method, mkt, cache; vols = reshape(v, nT, nK)), vec(mkt.Volatility))
            @test count(==(-1), J) == length(e) && count(iszero, J) == length(J) - length(e)
            @test all(==(-1), sum(J; dims = 2))
            @test count(iszero, sum(J; dims = 1)) == 9
            # a price that is not positive is a DomainError: the loss is Inf there
            @test_throws DomainError vol_errors([0.2], model, und, MonteCarloConfiguration(10, 1), mkt)
            @test FC.loss([0.2], model, und, MonteCarloConfiguration(10, 1), mkt) == Inf
            # parameters the model rejects (a DomainError of its constructor) too; any other exception propagates (#3)
            @test FC.loss([-0.2], model, und, method, mkt) == Inf
            @test_throws MethodError FC.loss([0.2], model, und, NoPricer(), mkt)
            # where only the derivatives fail, the values are priced: the exception propagates instead of a zero
            # gradient (#42); where the values cannot be priced, still Inf and a zero gradient
            @test FC.loss([0.2], model, und, DerivativeFails(), mkt) == FC.loss([0.2], model, und, AnalyticMethod(), mkt)
            @test_throws DomainError ForwardDiff.gradient(x -> FC.loss(x, model, und, DerivativeFails(), mkt), [0.2])
            @test ForwardDiff.gradient(x -> FC.loss(x, model, und, DerivativeFails(), mkt), [-0.2]) == [0.0]
            @test FC.primal_value(ForwardDiff.Dual(ForwardDiff.Dual(0.2, 1.0), 1.0)) === 0.2
            # Zygote differentiates the volatility errors and the loss (its try through a rule, #40) of a closed form: the prices
            # are one allocating broadcast, not a write into an array (#39)
            slnm = ShiftedProcess(LogNormalMixture((0.1, 0.2, 0.35), (0.3, 0.4)), -0.1)
            p = [0.1, 0.2, 0.35, 0.3, 0.4, -0.1]
            scache = FC.loss_cache(slnm, und, AnalyticMethod(), mkt)
            half_squares(x) = sum(abs2, vol_errors(x, slnm, AnalyticMethod(), mkt, scache)) / 2
            reference = ForwardDiff.gradient(half_squares, p)
            @test Zygote.gradient(half_squares, p)[1] ≈ reference rtol = 1e-12
            @test Zygote.gradient(x -> FC.loss(x, slnm, AnalyticMethod(), mkt, scache), p)[1] ≈ reference rtol = 1e-12
        end

        # The Levenberg–Marquardt polish ends where the gradient of the loss vanishes, on the market surface too.
        @testset "Stationary on the market" begin
            progress("Stationary on the market")
            model = BlackScholesProcess(0.3)
            method = CosMethod(1024)
            res = calibrate(model, und, method, mkt)
            g = ForwardDiff.gradient(p -> FC.loss(p, model, und, method, mkt), res.OptimalParam)
            @test norm(g) < 1e-9
            # without it, NelderMead stops at its own tolerances
            @test calibrate(model, und, method, mkt; polish = false).OptimalValue >= res.OptimalValue
            # a start point that cannot be priced (a too coarse Lewis gives a negative price): the Inf result, with and
            # without the polish (#38; with it, calibrate used to throw the DomainError of the loss)
            unpriceable(polish) = calibrate(HestonProcess(0.3, 0.2, 0.0, 1.0, -0.5, 0.04), und, LewisMethod(100.0, 501), mkt; polish = polish).OptimalValue
            @test unpriceable(false) == unpriceable(true) == Inf
            # no model volatility at the quotes left out of the loss
            @test count(isnan, res.ModelVolatility) == 9
            # the model volatilities come from the errors of the polish: the same as pricing the surface again
            cache = FC.loss_cache(model, und, method, mkt)
            calibrated = FinancialMonteCarlo.set_parameters!(model, res.OptimalParam)
            @test all(res.ModelVolatility[i, cache.quoted[i]] ≈ FC.model_volatilities(calibrated, und.S0, mkt, cache, i) for i in eachindex(cache.quoted) if !isempty(cache.quoted[i]))
            @test res.OptimalValue ≈ FC.loss(res.OptimalParam, model, und, method, mkt)
        end

        # The default parametrizations: the inverse of each other on valid parameters, and every variable a valid model
        # (the constraints between the parameters of VG, NIG and the mixtures included), so the loss is finite (#24).
        @testset "Default parametrizations" begin
            progress("Default parametrizations")
            models = (BlackScholesProcess(0.2), MertonProcess(0.15, 0.3, -0.1, 0.2), KouProcess(0.15, 0.5, 0.4, 10.0, 15.0), VarianceGammaProcess(0.2, -0.03, 0.16), NormalInverseGaussianProcess(0.2, -0.03, 0.16), HestonProcess(0.3, 0.2, 0.0, 1.5, -0.6, 0.04), ShiftedProcess(LogNormalMixture((0.2, 0.25, 0.3), (0.2, 0.3)), -0.1), LogNormalMixture((0.2, 0.3), (0.4,)))
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
            progress("calibrated_parameters")
            model = BlackScholesProcess(0.3)
            method = CosMethod(1024)
            n = sum(length, FC.loss_cache(model, und, method, mkt).quoted)
            @test calibrated_parameters(model, und, method, mkt) ≈ calibrate(model, und, method, mkt).OptimalParam
            i = CartesianIndex(4, 5)
            bump(h) = (V = mkt.Volatility .+ zero(h); V[i] += h; V)
            ∂σ = ForwardDiff.derivative(h -> calibrated_parameters(model, und, method, mkt, bump(h)), 0.0)
            @test ∂σ ≈ [1 / n] rtol = 1e-8
            h = 1e-4
            @test ∂σ ≈ (calibrated_parameters(model, und, method, mkt, bump(h)) - calibrated_parameters(model, und, method, mkt, bump(-h))) / 2h rtol = 1e-4
            @test_throws DimensionMismatch calibrated_parameters(model, und, method, mkt, mkt.Volatility[:, 2:end])
        end

        @testset "Numerical methods agree" begin
            progress("Numerical methods agree")
            p0 = [0.2, -0.03, 0.16]
            model = NormalInverseGaussianProcess(0.2, -0.03, 0.16)
            args = (mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
            ref = smile_prices(model, und, CarrMadanMethod(400.0, 14), p0, args...)
            for m in (CarrMadanLewisMethod(400.0, 14), CosMethod(256))
                @test isapprox(smile_prices(model, und, m, p0, args...), ref; rtol = 1e-3)
            end
        end

        @testset "Shifted lognormal mixture" begin
            progress("Shifted lognormal mixture")
            Random.seed!(0)
            model = ShiftedProcess(LogNormalMixture(ntuple(_ -> 0.2, 4), Tuple(rand(3) / 4)), -0.22)
            method = AnalyticMethod()
            p0 = FinancialMonteCarlo.get_parameters(model)
            @test length(p0) == 8
            @test FC.loss(p0, model, und, method, mkt) < 1e4
        end
    end

    @testset "Float32" begin
        progress("Float32")
        raw = retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy())
        und, mkt = adapter(raw, Float32)
        @test und.S0 isa Float32
        @test mkt isa MarketData{Float32}
        @test eltype(mkt.Volatility) == Float32

        model = ShiftedProcess(LogNormalMixture(ntuple(_ -> 0.2f0, 4), ntuple(_ -> 0.1f0, 3)), -0.22f0)
        p0 = FinancialMonteCarlo.get_parameters(model)
        @test eltype(p0) == Float32
        prices = smile_prices(model, und, AnalyticMethod(), p0, mkt.Strikes, mkt.ZeroRates[1], mkt.TimeToMaturity[1])
        @test eltype(prices) == Float32
        @test FC.loss(p0, model, und, AnalyticMethod(), mkt) isa Float32

        nig = NormalInverseGaussianProcess(0.2f0, -0.03f0, 0.16f0)
        method = CarrMadanMethod(400.0f0, 14, 0.75f0)
        @test FC.loss(FinancialMonteCarlo.get_parameters(nig), nig, und, method, mkt) isa Float32
    end

    @testset "AnalyticMethod on a model without closed form" begin
        progress("AnalyticMethod on a model without closed form")
        und = FinancialMonteCarlo.Underlying(100.0)
        model = NormalInverseGaussianProcess(0.2, -0.03, 0.16)
        @test_throws MethodError smile_prices(model, und, AnalyticMethod(), [0.2, -0.03, 0.16], [100.0], 0.01, 1.0)
    end
end
progress("Finished the core testset")

logged_include("models_methods.jl")
logged_include("financialfft_example.jl")
logged_include("cboe.jl")
# market data that is not in the repository: run where it is on disk
if isfile(joinpath(@__DIR__, "..", "data", "json", "cboe", "2026-11-12", "CboeOptions_SPX12Nov2026.json"))
    logged_include(joinpath("local", "cboe_spx.jl"))
else
    @info "No SPX option chain in data/json/cboe/2026-11-12: skipping the local tests"
end
logged_include("context.jl")
logged_include("aqua.jl")

using CUDA
if CUDA.functional()
    logged_include("gpu.jl")
else
    @info "No functional GPU: skipping the GPU tests"
end
