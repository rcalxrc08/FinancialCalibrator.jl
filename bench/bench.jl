# One case per process:  julia --project=. bench.jl SCENARIO ARRAYS PART ARGS...
#   SCENARIO  slnm (ShiftedProcess(LogNormalMixture), AnalyticMethod, 4 parameters)
#             vg   (VarianceGamma, LewisMethod(100, 4001), 3 parameters)
#             heston (Heston, CosMethod(1024), 4 free parameters, λ fixed)
#             big  (Heston, CosMethod(1024), 4 maturities × 500 strikes)
#   ARRAYS    dax64 (host, Float64, the DAX underlying with its dividend curve; not for big)
#             host64 | host32 | dev64 | dev32  (flat dividend, so that the same data go to the GPU)
#   PART      A <backend>                   gradient of loss and Jacobian of vol_errors in the variables
#             B <autodiff> <polish> <opt>   calibrate(…, opt; autodiff, polish_backend), opt NM or LBFGS
#             C <outer> <conditions> <x> <y> gradient of w' calibrated_parameters(v)
#             J <backend>                   full Jacobian ∂p*/∂v
# Quotes are synthetic (a known model on the DAX strikes and maturities): a well-conditioned optimum.
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT, Optim, LinearAlgebra, Printf, Dates
import ForwardDiff, Zygote, Mooncake, Enzyme
import DifferentiationInterface as DI
using ADTypes
const FC = FinancialCalibrator;
const FMC = FinancialMonteCarlo
const scen, arrays, part = ARGS[1], ARGS[2], ARGS[3]
const device = startswith(arrays, "dev")
if device
    using CUDA, Adapt
end
const T = endswith(arrays, "32") ? Float32 : Float64
backend(name) = name == "FD" ? AutoForwardDiff() : name == "Zygote" ? AutoZygote() : name == "Mooncake" ? AutoMooncake(; config = nothing) : name == "EnzymeR" ? AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const) : name == "EnzymeF" ? AutoEnzyme(; mode = Enzyme.Forward, function_annotation = Enzyme.Const) : error(name)
sync() = device ? CUDA.synchronize() : nothing

const und_dax, mkt_dax = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
const und = arrays == "dax64" ? und_dax : Underlying(T(und_dax.S0), T(FC.implied_dividend(und_dax, 1.0)))
function build(::Type{T}, und) where {T}
    grid = if scen == "big"
        Ts = [0.5, 1.0, 1.5, 2.0]
        MarketData(T.(Ts), [Date(2016, 8, 22) + Day(round(Int, 365t)) for t in Ts], fill(T(0.0), 4), collect(range(T(0.6 * und.S0), T(1.6 * und.S0); length = 500)), fill(T(0.2), 4, 500))
    else
        MarketData(T.(mkt_dax.TimeToMaturity), mkt_dax.MaturityDates, T.(mkt_dax.ZeroRates), T.(mkt_dax.Strikes), T.(mkt_dax.Volatility))
    end
    method, truth, model = if scen == "slnm"
        AnalyticMethod(), ShiftedProcess(LogNormalMixture(T.((0.15, 0.3)), T.((0.6,)), und), T(0.1)), ShiftedProcess(LogNormalMixture(T.((0.2, 0.35)), T.((0.5,)), und), T(0.05))
    elseif scen == "vg"
        LewisMethod(T(100), 4001), VarianceGammaProcess(T(0.2), T(-0.1), T(0.2), und), VarianceGammaProcess(T(0.25), T(-0.05), T(0.3), und)
    else
        CosMethod(1024), HestonProcess(T(0.3), T(0.2), T(0), T(1.5), T(-0.6), T(0.04), und), HestonProcess(T(0.25), T(0.25), T(0), T(1.0), T(-0.5), T(0.05), und)
    end
    # the synthetic surface, built on the host
    mkt = let c = FC.loss_cache(truth, method, grid), v = copy(grid.Volatility)
        v[c.flat.positions] .= FC.vol_errors(FMC.get_parameters(truth), truth, method, grid, c; vols = zero(v))
        MarketData(grid.TimeToMaturity, grid.MaturityDates, grid.ZeroRates, grid.Strikes, v)
    end
    return method, truth, model, mkt
end
const method, truth, model, mkt_host = build(T, und)
const mkt = device ? Adapt.adapt(CuArray, mkt_host) : mkt_host
# the Float64 host problem with the same data, for the references of C and J
const und64 = arrays == "dax64" ? und_dax : Underlying(Float64(und.S0), Float64(FC.implied_dividend(und, 1.0)))
const methodh, _, modelh, mh = build(Float64, und64)
fmt(t) = t < 1e-3 ? @sprintf("%.1f μs", 1e6t) : t < 1 ? @sprintf("%.2f ms", 1e3t) : @sprintf("%.2f s", t)
function timed(f; n = 0)
    first_ = @elapsed (r = f(); sync())
    second = @elapsed (f(); sync())
    n == 0 && (n = second < 0.005 ? 200 : second < 0.2 ? 20 : second < 2 ? 5 : 2)
    best = min(second, minimum(@elapsed((f(); sync())) for _ = 1:n))
    return r, first_, best
end
failmsg(e) = "FAIL " * replace(first(split(sprint(showerror, e), '\n')), '|' => '/')[1:min(end, 110)]
const p = FC.default_parametrization(model)
const x0 = p.from_params(FMC.get_parameters(model))
const cache = FC.loss_cache(model, method, mkt)
const nq = length(cache.flat.positions)
tag = "$scen | $arrays | $part | nq $nq"

if part == "A"
    b = backend(ARGS[4])
    L(x) = FC.loss(p.to_params(x), model, method, mkt, cache)
    E(x) = FC.vol_errors(p.to_params(x), model, method, mkt, cache)
    gref = ForwardDiff.gradient(L, x0)
    Jref = Array(ForwardDiff.jacobian(E, x0))
    _, _, tv = timed(() -> L(x0))
    for (what, f, ref, op) in (("loss gradient", L, gref, DI.gradient), ("vol_errors Jacobian", E, Jref, DI.jacobian))
        res = try
            prep = what == "loss gradient" ? DI.prepare_gradient(f, b, x0) : DI.prepare_jacobian(f, b, x0)
            g, t1, t = timed(() -> op(f, prep, b, x0))
            err = norm(Array(g) - ref) / norm(ref)
            @sprintf("first %s | run %s | err %.0e", fmt(t1), fmt(t), err)
        catch e
            failmsg(e)
        end
        println("RES | $tag | ", ARGS[4], " | ", what, " | ", res, " | value ", fmt(tv))
    end
elseif part == "B"
    a, pb = backend(ARGS[4]), backend(ARGS[5])
    opt = ARGS[6] == "NM" ? NelderMead() : LBFGS()
    res = try
        cal, t1, t = timed(() -> calibrate(model, method, mkt, opt; autodiff = a, polish_backend = pb); n = 3)
        perr = norm(cal.OptimalParam - FMC.get_parameters(truth)) / norm(FMC.get_parameters(truth))
        @sprintf("first %s | run %s | loss %.1e | param err %.0e", fmt(t1), fmt(t), cal.OptimalValue, perr)
    catch e
        failmsg(e)
    end
    println("RES | $tag | autodiff ", ARGS[4], " polish ", ARGS[5], " ", ARGS[6], " | ", res)
elseif part in ("C", "J")
    v0 = copy(mkt.Volatility)
    np = length(FMC.get_parameters(model))
    w = T.(collect(1:np))
    # the reference, on the Float64 host problem: ForwardDiff with one chunk, Zygote outer on the big surface
    vh = mh.Volatility
    fh(v) = calibrated_parameters(modelh, methodh, mh, v)
    big = scen == "big"
    calibrated_parameters(model, method, mkt, v0)   # one calibration: the first call below is the compilation of the derivatives
    if part == "C"
        outer, cb, xb, yb = ARGS[4], backend(ARGS[5]), backend(ARGS[6]), backend(ARGS[7])
        f(v) = dot(w, calibrated_parameters(model, method, mkt, v; conditions_backend = cb, implicit_backends = (x = xb, y = yb)))
        gref = big ? Zygote.gradient(v -> dot(Float64.(w), fh(v)), vh)[1] : ForwardDiff.gradient(v -> dot(Float64.(w), fh(v)), vh, ForwardDiff.GradientConfig(nothing, vh, ForwardDiff.Chunk{length(vh)}()))
        res = try
            grad = outer == "FD" ? (() -> ForwardDiff.gradient(f, v0, ForwardDiff.GradientConfig(f, v0, ForwardDiff.Chunk{length(v0)}()))) : (() -> DI.gradient(f, backend(outer), v0))
            g, t1, t = timed(grad; n = 3)
            @sprintf("first %s | run %s | err %.0e", fmt(t1), fmt(t), norm(Array(g) - gref) / norm(gref))
        catch e
            failmsg(e)
        end
        println("RES | $tag | outer ", ARGS[4], " cond ", ARGS[5], " x ", ARGS[6], " y ", ARGS[7], " | ", res)
    else
        Jref = big ? Zygote.jacobian(fh, vh)[1] : ForwardDiff.jacobian(fh, vh, ForwardDiff.JacobianConfig(nothing, vh, ForwardDiff.Chunk{length(vh)}()))
        bname = ARGS[4]
        pstar(v) = calibrated_parameters(model, method, mkt, v)
        res = try
            jac = bname == "FD" ? (() -> ForwardDiff.jacobian(pstar, v0, ForwardDiff.JacobianConfig(pstar, v0, ForwardDiff.Chunk{length(v0)}()))) : (() -> DI.jacobian(pstar, backend(bname), v0))
            J, t1, t = timed(jac; n = 3)
            @sprintf("first %s | run %s | err %.0e", fmt(t1), fmt(t), norm(Array(J) - Jref) / norm(Jref))
        catch e
            failmsg(e)
        end
        println("RES | $tag | jacobian ", bname, " | ", res)
    end
end
