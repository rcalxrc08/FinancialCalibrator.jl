# The figures and the key numbers of the guides (calibration.md, closed_form.md, exotics.md), included by figures.jl
# after home.jl (so THEMES, VIZ, base_plot, save_themes, tile and tenor are in scope). The numbers are HTML tiles
# returned by functions that a hidden @example block of each page calls with the variables of that page; the figures
# are drawn from the @example blocks of the page, run in a module of their own (page_module).
using LinearAlgebra

tiles(items...) = HTML("<div class=\"fc-stats\">" * join(items) * "</div>")
sci(x) = (e = floor(Int, log10(abs(x))); @sprintf("%.1f×10<sup>%d</sup>", x / 10.0^e, e))

# ---------- Calibration ----------

"The fit of the Heston calibration of calibration.md, and how stationary its end point is."
function calibration_tiles(res, vols, start, method, mkt, variables)
    cache = FinancialCalibrator.loss_cache(start, method, mkt; min_vega = 0)
    x = variables.from_params(res.OptimalParam)
    g = ForwardDiff.gradient(y -> FinancialCalibrator.loss(variables.to_params(y), start, method, mkt, cache), x)
    e = filter(!isnan, res.ModelVolatility .- vols)
    return tiles(tile(@sprintf("%.1f", 1e4 * sqrt(sum(abs2, e) / length(e))), " bp", "RMS volatility error, quotes moved by up to 1% in price"), tile(@sprintf("%.0f", 1e4 * maximum(abs, e)), " bp", "largest error, over $(length(e)) quotes"; tone = "is-amber"), tile(sci(norm(g)), "", "|∇L| at the end point: the polish reaches a stationary point"; tone = "is-teal"), tile("$(length(x))", " variables", "searched, unconstrained; λ fixed at 0 by the parametrization"))
end

const VARIABLE_NAMES = ["log σ", "log σ₀", "log κ", "artanh ρ", "log θ"]
function identifiability_figure(theme, F)
    v = viz(theme)
    common = (legend = false, ygrid = false, framestyle = :axes, yticks = (1:5, VARIABLE_NAMES), yflip = true, ylims = (0.4, 5.6))
    a = base_plot(theme; title = "Singular values of J (log scale)", xscale = :log10, common..., yticks = (1:5, ["s₁", "s₂", "s₃", "s₄", "s₅"]), left_margin = 6Plots.mm)
    bar!(a, 1:5, F.S; orientation = :horizontal, bar_width = 0.6, color = [k == 5 ? v.series[3] : v.series[1] for k = 1:5], linewidth = 0, fillrange = minimum(F.S) / 3)
    w = F.V[:, end] .* sign(F.V[argmax(abs.(F.V[:, end])), end])
    b = base_plot(theme; title = "Weakest direction (vector of s₅)", common..., xlims = (-1, 1))
    bar!(b, 1:5, w; orientation = :horizontal, bar_width = 0.6, color = [x >= 0 ? v.series[3] : v.series[1] for x in w], linewidth = 0)
    vline!(b, [0]; color = theme.axis, linewidth = 1)
    return plot(a, b; layout = (1, 2), size = (960, 330), background_color = :transparent)
end

# ∂p*/∂v of the page: one heatmap per free parameter, maturity by strike
function implicit_figure(theme, ∂p_∂v, maturities, strikes)
    v = viz(theme)
    names = ["σ", "σ₀", "λ", "κ", "ρ", "θ"]
    panels = map([1, 2, 4, 5, 6]) do k
        s = reshape(∂p_∂v[k, :], length(maturities), length(strikes))
        m = maximum(abs, s)
        p = base_plot(theme; title = "∂$(names[k])*/∂v", titlefontsize = 11, framestyle = :box, grid = false, left_margin = 2Plots.mm, bottom_margin = 2Plots.mm)
        heatmap!(p, 1:length(strikes), 1:length(maturities), s; color = cgrad(v.diverging), clims = (-m, m), xticks = ([1, 7, 13], string.(Int.(strikes[[1, 7, 13]]))), yticks = (1:length(maturities), string.(maturities)))
        p
    end
    note = plot(; framestyle = :none, background_color = :transparent, grid = false, ticks = false)
    annotate!(note, 0.05, 0.6, text("rows: maturity (years)\ncolumns: strike\n\namber: the parameter rises\nwith the quote; blue: it falls\n\neach panel has its own scale", 10, theme.muted, :left))
    xlims!(note, 0, 1)
    ylims!(note, 0, 1)
    return plot(panels..., note; layout = (2, 3), size = (960, 560), background_color = :transparent)
end

# ---------- Closed forms ----------

"Black–Scholes against the shifted lognormal mixture on the DAX, with the closed form."
function closed_form_tiles(bs, slnm)
    rms(res) = 1e4 * sqrt(2 * res.OptimalValue / count(!isnan, res.ModelVolatility))
    return tiles(tile(@sprintf("%.0f", rms(bs)), " bp", "RMS error, Black–Scholes: one volatility for the whole surface"), tile(@sprintf("%.0f", rms(slnm)), " bp", "RMS error, three-component shifted lognormal mixture"; tone = "is-teal"), tile(@sprintf("%.1f", rms(bs) / rms(slnm)), "×", "smaller error, with six parameters and no numerical method"; tone = "is-amber"))
end

function closed_form_figure(theme, mkt, S0, bs, slnm)
    v = viz(theme)
    rows = [1, 3, 5, 9]
    panels = map(rows) do i
        p = base_plot(theme; title = "T = $(tenor(mkt.TimeToMaturity[i]))", titlefontsize = 10, legend = i == rows[1] ? :topright : false, ylims = (0.1, 0.36), yticks = (0.15:0.05:0.35, ["15%", "20%", "25%", "30%", "35%"]), xticks = 0.8:0.2:1.6, left_margin = 2Plots.mm, bottom_margin = 2Plots.mm)
        k = mkt.Strikes ./ S0
        plot!(p, k, slnm.ModelVolatility[i, :]; color = v.series[2], linewidth = 2, marker = :circle, markersize = 3, markerstrokewidth = 0, label = "mixture")
        plot!(p, k, bs.ModelVolatility[i, :]; color = v.series[1], linewidth = 2, linestyle = :dash, label = "Black–Scholes")
        scatter!(p, k, mkt.Volatility[i, :]; color = v.series[3], markersize = 4.5, markerstrokewidth = 0, label = "quotes")
        p
    end
    return plot(panels...; layout = (2, 2), size = (960, 560), background_color = :transparent)
end

# ---------- Exotics ----------

"The bucketed vega of the position of exotics.md."
function exotic_tiles(∂V_∂v, maturities, strikes)
    i = argmax(abs.(∂V_∂v))
    return tiles(tile("$(length(∂V_∂v))", " quotes", "sensitivities from one gradient, through Monte Carlo and the calibration"), tile(@sprintf("%.1f", sum(∂V_∂v)), "", "sensitivity to a parallel shift of the surface: the sum of the buckets"; tone = "is-teal"), tile(@sprintf("%.1f", ∂V_∂v[i]), "", "largest bucket: the quote at T = $(maturities[i[1]]), K = $(Int(strikes[i[2]]))"; tone = "is-amber"))
end

let calibration = CALIBRATION_PAGE
    F, ∂p_∂v, maturities, strikes = Base.invokelatest(() -> (calibration.F, calibration.∂p_∂v, calibration.maturities, calibration.strikes))
    save_themes(theme -> identifiability_figure(theme, F), "identifiability")
    save_themes(theme -> implicit_figure(theme, ∂p_∂v, maturities, strikes), "implicit")
end
let closed_form = page_module("closed_form.md", "closed_form", :slnm)
    data = Base.invokelatest(() -> (closed_form.mkt, closed_form.underlying.S0, closed_form.bs, closed_form.slnm))
    save_themes(theme -> closed_form_figure(theme, data...), "closed-form")
end
