# The figures and the numbers of the home page, included by figures.jl (so THEMES, MARKERS, base_plot and save_themes
# are in scope). Everything drawn here comes from the package: a Heston model calibrated to the bundled DAX surface,
# the implicit derivatives of that calibration, and the benchmarks measured by docs/benchmarks.jl
# (src/assets/benchmarks.toml). The HTML fragments (stat tiles, benchmark cards) are returned by functions that the
# hidden @example blocks of index.md call.
using FinancialCalibrator, FinancialMonteCarlo, FinancialFFT, FinancialToolbox, ForwardDiff, Dates, Printf, TOML

# The visualisation palette of the site: categorical slots in fixed order (blue, teal, amber, cyan, green), a
# sequential ramp for magnitudes (volatility, loss) and a diverging pair (blue below zero, amber above) around the
# surface colour. The light variants are darker steps of the same hues.
const VIZ = (dark = (series = ["#4f8cff", "#00d4aa", "#ffb84d", "#38cfff", "#7ee787"], ramp = ["#22346b", "#4f8cff", "#38cfff", "#00d4aa", "#7ee787", "#ffb84d"], diverging = ["#4f8cff", "#151b2e", "#ffb84d"], marker = "#ffb84d", ring = "#0b1020", text = "#f5f7fa", muted = "#a0aec0", axis = "#3a4466", grid = "#252d47", floor = "#151b2e"), light = (series = ["#2563d9", "#0a8a6f", "#b86e0b", "#0b7fb3", "#2f8a3c"], ramp = ["#c7d6f5", "#4f8cff", "#1ea6d4", "#00a585", "#4db55c", "#e8962a"], diverging = ["#2563d9", "#f7f6f3", "#c47a12"], marker = "#b86e0b", ring = "#ffffff", text = "#1f2328", muted = "#5b6470", axis = "#b9b4ab", grid = "#e6e3de", floor = "#f3f1ec"))
viz(theme) = theme.text == THEMES.dark.text ? VIZ.dark : VIZ.light

# ---------- The calibration shown on the home page ----------

const DAX = let
    und, mkt = adapter(retriever(CalibrationID("Dax", "22Aug2016", "EUR"), IOFileInputStrategy()))
    method = LewisMethod(100.0, 4001)
    model = HestonProcess(0.3, 0.2, 0.0, 1.0, -0.5, 0.04, und)
    res = calibrate(model, method, mkt)
    (und = und, mkt = mkt, method = method, model = model, res = res, calibrated = FinancialMonteCarlo.set_parameters!(model, res.OptimalParam))
end

# The implied volatilities of the calibrated model on a grid of strikes and maturities, computed as the loss computes
# them (out of the money smiles of `pricer`, FinancialToolbox's blsimpv): a MarketData on the grid, whose cache gives the
# smiles. Point by point, so that a far wing priced at zero leaves a gap (NaN) instead of the whole maturity.
function model_surface(d, strikes, maturities)
    S0 = d.und.S0
    rates = [FinancialCalibrator.interpolate_linear(d.mkt.TimeToMaturity, d.mkt.ZeroRates, t) for t in maturities]
    grid = MarketData(maturities, [Date(2016, 8, 22) + Day(round(Int, 365 * t)) for t in maturities], rates, strikes, fill(0.25, length(maturities), length(strikes)))
    cache = FinancialCalibrator.loss_cache(d.calibrated, d.method, grid; min_vega = 0)
    vols = fill(NaN, length(maturities), length(strikes))
    for i in eachindex(maturities)
        prices = vec(pricer(d.calibrated, cache.zero_rates[i], cache.method, cache.smiles[i]))
        for (n, j) in enumerate(cache.quoted[i])
            prices[n] > 3e-5 * S0 || continue                  # below the accuracy of the quadrature
            vols[i, j] = try
                blsimpv(S0, strikes[j], rates[i], maturities[i], prices[n], cache.dividends[i], cache.is_call[i][n])
            catch e
                e isa DomainError || rethrow()
                NaN
            end
        end
    end
    return vols
end

rms_bp(res) = 1e4 * sqrt(2 * res.OptimalValue / count(!isnan, res.ModelVolatility))

# ---------- The hero: the calibrated surface and the quotes, in perspective, as an SVG of its own ----------

ramp_color(stops, u) = begin
    u = clamp(u, 0, 1) * (length(stops) - 1)
    i = min(floor(Int, u), length(stops) - 2)
    w = u - i
    a, b = hex_rgb(stops[i+1]), hex_rgb(stops[i+2])
    rgb_hex((1 - w) .* a .+ w .* b)
end
hex_rgb(h) = (parse(Int, h[2:3]; base = 16), parse(Int, h[4:5]; base = 16), parse(Int, h[6:7]; base = 16))
rgb_hex(c) = "#" * join(string(round(Int, clamp(x, 0, 255)); base = 16, pad = 2) for x in c)
mix(a, b, w) = rgb_hex((1 - w) .* hex_rgb(a) .+ w .* hex_rgb(b))

function surface_svg(theme, d, strikes, maturities, vols)
    v = viz(theme)
    S0 = d.und.S0
    lm = log.(strikes ./ S0)
    zlo = 0.12
    zhi = ceil(max(maximum(filter(isfinite, vols)), maximum(d.mkt.Volatility)) / 0.02) * 0.02
    X(k) = (log(k / S0) - first(lm)) / (last(lm) - first(lm)) - 0.5
    Y(t) = (t - first(maturities)) / (last(maturities) - first(maturities)) - 0.5
    Z(s) = 0.8 * (s - zlo) / (zhi - zlo)
    az, el = deg2rad(-38), deg2rad(24)
    W, H, scale, cx, cy = 960, 640, 600, 470, 440
    function project(x, y, z)
        xr = x * cos(az) - y * sin(az)
        yr = x * sin(az) + y * cos(az)
        return (cx + scale * xr, cy - scale * (z * cos(el) + yr * sin(el)), yr * cos(el) - z * sin(el))
    end
    pt(p) = @sprintf("%.1f,%.1f", p[1], p[2])
    items = Tuple{Float64, String}[]                       # (depth, svg): drawn from the farthest
    light = [-0.35, -0.55, 0.75] ./ sqrt(0.35^2 + 0.55^2 + 0.75^2)
    for i = 1:(length(maturities)-1), j = 1:(length(strikes)-1)
        z4 = (vols[i, j], vols[i, j+1], vols[i+1, j+1], vols[i+1, j])
        all(isfinite, z4) || continue
        c = [(X(strikes[j]), Y(maturities[i]), Z(z4[1])), (X(strikes[j+1]), Y(maturities[i]), Z(z4[2])), (X(strikes[j+1]), Y(maturities[i+1]), Z(z4[3])), (X(strikes[j]), Y(maturities[i+1]), Z(z4[4]))]
        p = [project(q...) for q in c]
        a, b = c[3] .- c[1], c[4] .- c[2]
        n = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
        n = n ./ sqrt(sum(abs2, n))
        shade = 0.72 + 0.28 * abs(sum(n .* light))
        fill = mix(v.floor, ramp_color(v.ramp, (sum(z4) / 4 - zlo) / (zhi - zlo)), shade)
        poly = "<polygon points=\"$(join(pt.(p), " "))\" fill=\"$fill\" stroke=\"$(mix(fill, v.text, 0.18))\" stroke-width=\"0.6\" stroke-opacity=\"0.55\"/>"
        push!(items, (sum(q[3] for q in p) / 4, poly))
    end
    # the quotes: filled where they enter the loss, hollow where they are left out
    mkt, res = d.mkt, d.res
    for i in eachindex(mkt.TimeToMaturity), j in eachindex(mkt.Strikes)
        p = project(X(mkt.Strikes[j]), Y(mkt.TimeToMaturity[i]), Z(mkt.Volatility[i, j]))
        used = !isnan(res.ModelVolatility[i, j])
        dot = used ? "<circle cx=\"$(@sprintf("%.1f", p[1]))\" cy=\"$(@sprintf("%.1f", p[2]))\" r=\"5\" fill=\"$(v.marker)\" stroke=\"$(v.ring)\" stroke-width=\"1.8\"/>" : "<circle cx=\"$(@sprintf("%.1f", p[1]))\" cy=\"$(@sprintf("%.1f", p[2]))\" r=\"4.4\" fill=\"none\" stroke=\"$(v.marker)\" stroke-width=\"1.6\"/>"
        push!(items, (p[3] - 1e-3, dot))
    end
    sort!(items; by = first, rev = true)
    io = IOBuffer()
    print(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $W $H" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" role="img" aria-label="Implied volatility surface of a Heston model calibrated to the DAX quotes of 22 August 2016, with the quotes as dots">""")
    # the floor: a grid in moneyness and maturity, and the axes
    zf = Z(zlo)
    line(a, b; color = v.grid, w = 1) = "<line x1=\"$(@sprintf("%.1f", a[1]))\" y1=\"$(@sprintf("%.1f", a[2]))\" x2=\"$(@sprintf("%.1f", b[1]))\" y2=\"$(@sprintf("%.1f", b[2]))\" stroke=\"$color\" stroke-width=\"$w\"/>"
    label(p, s; anchor = "middle", color = v.muted, size = 17) = "<text x=\"$(@sprintf("%.1f", p[1]))\" y=\"$(@sprintf("%.1f", p[2]))\" fill=\"$color\" font-size=\"$size\" text-anchor=\"$anchor\">$s</text>"
    corners = [project(x, y, zf) for (x, y) in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
    print(io, "<polygon points=\"$(join(pt.(corners), " "))\" fill=\"$(v.floor)\" fill-opacity=\"0.55\" stroke=\"$(v.axis)\"/>")
    moneyness_ticks = [0.8, 1.0, 1.2, 1.4, 1.6]
    for m in moneyness_ticks
        x = X(m * S0)
        print(io, line(project(x, -0.5, zf), project(x, 0.5, zf)))
        print(io, label(project(x, -0.5, zf) .+ (-8, 28, 0), @sprintf("%.1f", m)))
    end
    for t in (0.5, 1.0, 2.0, 3.0, 4.0)
        y = Y(t)
        print(io, line(project(-0.5, y, zf), project(0.5, y, zf)))
        print(io, label(project(0.5, y, zf) .+ (14, 8, 0), t == 0.5 ? "6M" : "$(Int(t))Y"; anchor = "start"))
    end
    # the volatility axis on the far left corner
    base, top = project(0.5, 0.5, zf), project(0.5, 0.5, Z(zhi))
    print(io, line(base, top; color = v.axis))
    for s = 0.15:0.05:zhi
        q = project(0.5, 0.5, Z(s))
        print(io, line(q, q .+ (7, 0, 0); color = v.axis), label(q .+ (12, 6, 0), "$(round(Int, 100s))%"; anchor = "start"))
    end
    print(io, label(top .+ (0, -16, 0), "implied volatility"; anchor = "middle"))
    mid_front = project(0.0, -0.5, zf)
    print(io, label(mid_front .+ (-40, 66, 0), "moneyness K / S₀"), label(project(0.5, -0.25, zf) .+ (70, 40, 0), "maturity"; anchor = "start"))
    for (_, s) in items
        print(io, s)
    end
    # the legend is in the caption of the page (index.md), in HTML
    print(io, "</svg>")
    return String(take!(io))
end

# ---------- The figures of the "Results" section ----------

moneyness_label = "moneyness K / S₀"
tenor(t) = t < 1 ? "$(round(Int, 12t))M" : (isinteger(t) ? "$(Int(t))Y" : "$(t)Y")

function smiles_figure(theme, d, strikes, vols)
    v = viz(theme)
    mkt, res, S0 = d.mkt, d.res, d.und.S0
    panels = map(eachindex(mkt.TimeToMaturity)) do i
        p = base_plot(theme; title = "T = $(tenor(mkt.TimeToMaturity[i]))", titlefontsize = 10, legend = false, ylims = (0.12, 0.34), yticks = (0.15:0.05:0.30, ["15%", "20%", "25%", "30%"]), xticks = 0.8:0.2:1.6, left_margin = 2Plots.mm, bottom_margin = 2Plots.mm)
        plot!(p, strikes ./ S0, vols[i, :]; color = v.series[1], linewidth = 2)
        used = .!isnan.(res.ModelVolatility[i, :])
        scatter!(p, mkt.Strikes[used] ./ S0, mkt.Volatility[i, used]; color = v.series[3], markersize = 4.5, markerstrokewidth = 0)
        scatter!(p, mkt.Strikes[.!used] ./ S0, mkt.Volatility[i, .!used]; color = v.ring, markerstrokecolor = v.series[3], markerstrokewidth = 1.2, markersize = 4)
        p
    end
    return plot(panels...; layout = (3, 3), size = (960, 640), background_color = :transparent, plot_title = "Heston on the DAX: model smile (line) and quotes (dots), all nine maturities", plot_titlefontsize = 12, plot_titlefontcolor = theme.text, plot_titlelocation = :left)
end

function error_figure(theme, d)
    v = viz(theme)
    mkt, res = d.mkt, d.res
    e = 1e4 .* (res.ModelVolatility .- mkt.Volatility)
    m = maximum(abs, filter(isfinite, e))
    labels = [@sprintf("%.2f", k / d.und.S0) for k in mkt.Strikes]
    p = base_plot(theme; title = "Model minus quoted volatility, in basis points (blank: quote left out of the loss)", xlabel = moneyness_label, ylabel = "maturity", size = (960, 420), colorbar_title = "bp", framestyle = :box, grid = false)
    heatmap!(p, 1:length(mkt.Strikes), 1:length(mkt.TimeToMaturity), e; color = cgrad(v.diverging), clims = (-m, m), xticks = (1:length(labels), labels), yticks = (1:length(mkt.TimeToMaturity), tenor.(mkt.TimeToMaturity)))
    for i in axes(e, 1), j in axes(e, 2)
        isfinite(e[i, j]) && annotate!(p, j, i, text(string(round(Int, e[i, j])), 8, theme.text))
    end
    return p
end

# ∂p*/∂v of the calibrated Heston parameters, one heatmap per parameter over the surface: the implicit derivatives of
# calibrated_parameters, a 6 × 108 Jacobian from one ForwardDiff call.
const PARAMETER_NAMES = ["σ (vol of variance)", "σ₀ (initial vol)", "λ", "κ (mean reversion)", "ρ (correlation)", "θ (long-run variance)"]
function sensitivity_figure(theme, d, J)
    v = viz(theme)
    mkt, res = d.mkt, d.res
    left_out = isnan.(res.ModelVolatility)
    panels = map([5, 4, 6, 1, 2]) do k
        s = reshape(J[k, :], size(mkt.Volatility))
        s[left_out] .= NaN
        m = maximum(abs, filter(isfinite, s))
        p = base_plot(theme; title = "∂$(first(split(PARAMETER_NAMES[k])))*/∂v: $(PARAMETER_NAMES[k])", titlefontsize = 10, framestyle = :box, grid = false, left_margin = 2Plots.mm, bottom_margin = 3Plots.mm)
        heatmap!(p, 1:length(mkt.Strikes), 1:length(mkt.TimeToMaturity), s; color = cgrad(v.diverging), clims = (-m, m), xticks = ([1, 5, 9, 12], [@sprintf("%.2f", mkt.Strikes[j] / d.und.S0) for j in (1, 5, 9, 12)]), yticks = ([1, 3, 5, 7, 9], tenor.(mkt.TimeToMaturity[[1, 3, 5, 7, 9]])), colorbar = true)
        p
    end
    note = plot(; framestyle = :none, background_color = :transparent, grid = false, ticks = false)
    annotate!(note, 0.05, 0.6, text(
        "rows: maturity
columns: moneyness K / S₀

amber: the parameter rises
with the quote; blue: it falls

each panel has its own scale",
        10,
        theme.muted,
        :left
    ))
    xlims!(note, 0, 1)
    ylims!(note, 0, 1)
    return plot(panels..., note; layout = (2, 3), size = (1000, 600), background_color = :transparent)
end

# The loss over two parameters, the others at the optimum: the level sets of log₁₀ L in (κ, σ), whose long valley is
# the direction the quotes hardly see.
function landscape_figure(theme, d, κs, σs, L)
    v = viz(theme)
    p = d.res.OptimalParam
    q = base_plot(theme; title = "Level sets of log₁₀ L over κ and σ, the other parameters at the optimum", xlabel = "κ, mean reversion speed", ylabel = "σ, vol of variance", xscale = :log10, yscale = :log10, size = (960, 460), colorbar_title = "log₁₀ L", framestyle = :box, legend = false, xticks = ([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]), yticks = ([0.5, 1, 2], ["0.5", "1", "2"]))
    contour!(q, κs, σs, log10.(L)'; levels = 22, color = cgrad(v.ramp), linewidth = 1.4, fill = false)
    scatter!(q, [p[4]], [p[1]]; color = v.series[3], marker = :star5, markersize = 12, markerstrokewidth = 0)
    annotate!(q, p[4] * 1.12, p[1], text("optimum", 10, theme.text, :left))
    return q
end

# ---------- Benchmarks ----------

const BENCH = TOML.parsefile(joinpath(@__DIR__, "src", "assets", "benchmarks.toml"))

function optimizer_figure(theme)
    v = viz(theme)
    rows = BENCH["optimizers"]
    models = unique(r["model"] for r in rows)
    optimizers = unique(r["optimizer"] for r in rows)
    p = base_plot(theme; title = "Time to calibrate on the DAX surface, LewisMethod (median of 5 runs, polish included)", xlabel = "seconds", size = (960, 440), legend = :bottomright, yticks = (1:length(models), models), yflip = true, xgrid = true, ygrid = false, framestyle = :axes)
    w = 0.26
    xlims!(p, 0, 1.12 * maximum(r["seconds"] for r in rows))
    for (k, o) in enumerate(optimizers)
        t = [only(r["seconds"] for r in rows if r["model"] == m && r["optimizer"] == o) for m in models]
        bar!(p, (1:length(models)) .+ (k - 2) * w, t; orientation = :horizontal, bar_width = w * 0.86, color = v.series[k], linewidth = 0, label = o)
    end
    return p
end

function scaling_figure(theme)
    v = viz(theme)
    s = BENCH["scaling"]
    n = s["strikes"]
    p = base_plot(theme; title = "One loss on a smile of n strikes: Black–Scholes, CosMethod(4096)", xlabel = "strikes in the smile", ylabel = "milliseconds", xscale = :log10, yscale = :log10, size = (960, 440), legend = :topleft, xticks = (n, string.(n)), yticks = ([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"]))
    series = [("host_float64_ms", "CPU, Float64"), ("device_float64_ms", "GPU, Float64"), ("device_float32_ms", "GPU, Float32")]
    for (k, (key, name)) in enumerate(series)
        haskey(s, key) || continue
        plot!(p, n, s[key]; color = v.series[k], linewidth = 2, marker = MARKERS[k], markersize = 5, markerstrokewidth = 0, label = name)
    end
    return p
end

# The table of docs/src/backends.md (VG/Lewis, 15 quotes, a fresh process per case): the gradient of wᵀp*(v)
const AD_TABLE = [("ForwardDiff", 14.0, 0.20), ("Zygote", 29.0, 0.15), ("Mooncake", 30.0, 0.48), ("Enzyme, reverse", 87.0, 0.13), ("Enzyme, forward", 80.0, 0.12)]
function ad_figure(theme)
    v = viz(theme)
    names = first.(AD_TABLE)
    common = (yticks = (1:length(names), names), ylims = (0.4, length(names) + 0.6), yflip = true, legend = false, ygrid = false, framestyle = :axes)
    a = base_plot(theme; title = "First call: compilation (s)", common..., xlims = (0, 100), left_margin = 6Plots.mm)
    bar!(a, 1:length(names), getindex.(AD_TABLE, 2); orientation = :horizontal, bar_width = 0.6, color = v.series[1], linewidth = 0)
    b = base_plot(theme; title = "Next calls: one gradient (s)", common..., xlims = (0, 0.55))
    bar!(b, 1:length(names), getindex.(AD_TABLE, 3); orientation = :horizontal, bar_width = 0.6, color = v.series[2], linewidth = 0)
    return plot(a, b; layout = (1, 2), size = (960, 360), background_color = :transparent)
end

# ---------- HTML fragments for index.md ----------

fmt_s(t) = t < 1 ? @sprintf("%.0f ms", 1e3t) : @sprintf("%.2f s", t)
tile(value, unit, label; tone = "") = """<div class="fc-stat $tone"><div class="fc-stat-value">$value<span>$unit</span></div><div class="fc-stat-label">$label</div></div>"""

"The four numbers under the hero: the calibration that the hero shows."
function hero_stats()
    d = DAX
    heston_lbfgs = only(r for r in BENCH["optimizers"] if r["model"] == "Heston" && r["optimizer"] == "LBFGS")
    n = count(!isnan, d.res.ModelVolatility)
    tiles = [tile(@sprintf("%.0f", rms_bp(d.res)), " bp", "RMS volatility error, Heston on $(n) DAX quotes"), tile(@sprintf("%.2f", heston_lbfgs["seconds"]), " s", "to calibrate it with LBFGS + Levenberg–Marquardt, on a laptop CPU"; tone = "is-teal"), tile("6 × $(length(d.mkt.Volatility))", "", "Jacobian ∂p*/∂v from one ForwardDiff call"; tone = "is-amber"), tile("4", "", "AD backends through the calibration: ForwardDiff, Zygote, Mooncake, Enzyme")]
    return HTML("<div class=\"fc-stats\">" * join(tiles) * "</div>")
end

"The benchmark cards: the headline of each measurement, with the machine."
function benchmark_cards()
    rows = BENCH["optimizers"]
    best(m) = argmin(r -> r["seconds"], filter(r -> r["model"] == m, rows))
    nm(m) = only(r for r in rows if r["model"] == m && r["optimizer"] == "NelderMead")
    h = best("Heston")
    s = BENCH["scaling"]
    cards = String[]
    push!(cards, tile(fmt_s(h["seconds"]), "", "Heston on 99 DAX quotes with $(h["optimizer"]): $(@sprintf("%.1f", nm("Heston")["seconds"] / h["seconds"]))× faster than Nelder–Mead, same optimum ($(@sprintf("%.1f", h["rmse_bp"])) bp)"))
    push!(cards, tile("≈ 1", " calibration", "the cost of a full gradient ∂p*/∂v once compiled: the implicit derivatives themselves take ~8 ms"; tone = "is-teal"))
    if haskey(s, "device_float32_ms")
        k = length(s["strikes"])
        push!(cards, tile(@sprintf("%.1f", s["host_float64_ms"][k] / s["device_float32_ms"][k]), "×", "GPU in Float32 over the CPU, one loss on a smile of $(s["strikes"][k]) strikes ($(s["device"])); in Float64 the GPU overtakes the CPU near 500 strikes"; tone = "is-amber"))
    end
    push!(cards, tile("14 s", "", "first-call compilation with ForwardDiff outside; 29–30 s Zygote or Mooncake, 80–90 s Enzyme"))
    machine = "Measured $(BENCH["measured"]) on $(BENCH["cpu"]), $(BENCH["threads"]) thread(s), Julia $(BENCH["julia"])" * (haskey(s, "device") ? ", GPU $(s["device"])" : "") * ". Script: <code>docs/benchmarks.jl</code>; raw data: <code>docs/src/assets/benchmarks.toml</code>."
    return HTML("<div class=\"fc-stats fc-stats-bench\">" * join(cards) * "</div><p class=\"fc-machine\">$machine</p>")
end

"The engines table, from the benchmarks."
function engine_table()
    rows = BENCH["engines"]
    body = join(["<tr><td><code>$(r["engine"])</code></td><td>$(@sprintf("%.2f", r["loss_ms"])) ms</td><td>$(fmt_s(r["calibration_seconds"]))</td><td>$(@sprintf("%.1f", r["rmse_bp"])) bp</td></tr>" for r in rows])
    return HTML("<div class=\"fc-table-wrap\"><table class=\"fc-table\"><thead><tr><th>Heston, DAX, pricing engine</th><th>one loss</th><th>calibration</th><th>RMS error</th></tr></thead><tbody>$body</tbody></table></div><p class=\"fc-machine\">Same model, same optimum: the engine sets the cost. <code>CarrMadanLewisMethod(600, 16)</code> runs an FFT of 2¹⁶ points per maturity, far more than nine smiles of twelve strikes need.</p>")
end

# ---------- Draw ----------

let d = DAX, S0 = DAX.und.S0
    # the hero surface: fine in moneyness 0.8-1.75 and in maturity 2M-4Y
    strikes = collect(range(minimum(d.mkt.Strikes), maximum(d.mkt.Strikes); length = 46))
    maturities = collect(range(first(d.mkt.TimeToMaturity), last(d.mkt.TimeToMaturity); length = 34))
    surface = model_surface(d, strikes, maturities)
    for (name, theme) in pairs(THEMES)
        write(joinpath(FIGURES, "hero-surface-$name.svg"), surface_svg(theme, d, strikes, maturities, surface))
    end
    smile_strikes = collect(range(minimum(d.mkt.Strikes), maximum(d.mkt.Strikes); length = 120))
    smiles = model_surface(d, smile_strikes, d.mkt.TimeToMaturity)
    save_themes(theme -> smiles_figure(theme, d, smile_strikes, smiles), "home-smiles")
    save_themes(theme -> error_figure(theme, d), "home-errors")
    J = ForwardDiff.jacobian(w -> calibrated_parameters(d.model, d.method, d.mkt, w), d.mkt.Volatility)
    save_themes(theme -> sensitivity_figure(theme, d, J), "home-sensitivities")
    p = d.res.OptimalParam
    cache = FinancialCalibrator.loss_cache(d.model, d.method, d.mkt)
    κs = exp.(range(log(p[4] / 6), log(p[4] * 4); length = 44))
    σs = exp.(range(log(p[1] / 3), log(p[1] * 2.5); length = 44))
    L = [FinancialCalibrator.loss([σ, p[2], p[3], κ, p[5], p[6]], d.model, d.method, d.mkt, cache) for κ in κs, σ in σs]
    L[.!isfinite.(L)] .= NaN
    save_themes(theme -> landscape_figure(theme, d, κs, σs, L), "home-landscape")
    save_themes(optimizer_figure, "bench-optimizers")
    save_themes(scaling_figure, "bench-scaling")
    save_themes(ad_figure, "bench-ad")
end
