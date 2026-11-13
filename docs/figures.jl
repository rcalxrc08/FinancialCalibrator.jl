# Figures of the documentation, computed with the package at build time and saved as SVG in src/assets/figures, in a
# light and a dark variant: the stylesheet assets/figures.css shows the one of the theme of the reader.
using Plots
# GR draws without a display on the CI runner
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
gr()

const FIGURES = mkpath(joinpath(@__DIR__, "src", "assets", "figures"))

# Categorical slots in fixed order (validated for colour vision deficiencies on adjacent pairs, in both modes), and
# the ink of the text, of the axes and of the grid: the palette of the figures of FinancialFFT.jl.
const THEMES = (light = (series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], text = "#1f2328", muted = "#5b6470", axis = "#b9b4ab", grid = "#e6e3de", diverging = ["#eb6834", "#f7f6f3", "#2a78d6"]), dark = (series = ["#4f8cff", "#ff8a5c", "#00d4aa", "#ffb84d", "#ff6b9d", "#7ee081"], text = "#f5f7fa", muted = "#a0aec0", axis = "#3a4466", grid = "#252d47", diverging = ["#ff8a5c", "#151b2e", "#4f8cff"]))
# The secondary encoding of the series: the light slots 3-5 are below 3:1 contrast on the surface.
const MARKERS = [:circle, :rect, :utriangle, :diamond, :dtriangle, :star5]

function base_plot(theme; kwargs...)
    return plot(; size = (760, 430), background_color = :transparent, foreground_color_text = theme.muted, foreground_color_guide = theme.muted, foreground_color_axis = theme.axis, foreground_color_border = theme.axis, foreground_color_legend = nothing, legend_background_color = :transparent, legendfontcolor = theme.text, titlefontcolor = theme.text, gridcolor = theme.grid, gridalpha = 1, gridlinewidth = 1, framestyle = :axes, fontfamily = "sans-serif", titlefontsize = 12, guidefontsize = 10, tickfontsize = 9, legendfontsize = 9, titlelocation = :left, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm, kwargs...)
end

function save_themes(draw, name)
    for (theme_name, theme) in pairs(THEMES)
        savefig(draw(theme), joinpath(FIGURES, "$name-$theme_name.svg"))
    end
end

# The @example blocks of a page, run in a module of their own up to the one that defines `last_name`, so that a figure
# shows the data of the page without a copy of its code. The bindings are newer than this function: they are read in
# the latest world.
function page_module(page_name, block_name, last_name)
    page = read(joinpath(@__DIR__, "src", page_name), String)
    example = Module(Symbol(block_name))
    for block in eachmatch(Regex("```@example $block_name\\r?\\n(.*?)```", "s"), page)
        Base.include_string(example, block.captures[1])
        isdefined(example, last_name) && break
    end
    return example
end

# The implied volatilities of the quotes (markers) and of the calibrated model (lines), one series per maturity.
function calibration_figure(theme, (maturities, strikes, market_vols, model_vols))
    p = base_plot(theme; title = "Heston calibrated on four smiles: quotes (markers) and model (lines)", xlabel = "strike", ylabel = "implied volatility", legend = :outerright, size = (900, 430))
    for (j, T) in enumerate(maturities)
        plot!(p, strikes, model_vols[j, :]; label = nothing, color = theme.series[j], linewidth = 2)
        scatter!(p, strikes, market_vols[j, :]; label = "T = $T", color = theme.series[j], marker = MARKERS[j], markersize = 5, markerstrokewidth = 0)
    end
    return p
end

# ∂V/∂v over the surface: a diverging map centred at zero, one cell per quote, its value written in it.
function exotic_figure(theme, (maturities, strikes, sensitivities))
    m = maximum(abs, sensitivities)
    p = base_plot(theme; title = "Sensitivity of the position to each quoted volatility", xlabel = "strike", ylabel = "maturity", size = (900, 360), colorbar_title = "∂V/∂v", yflip = false)
    heatmap!(p, 1:length(strikes), 1:length(maturities), sensitivities; color = cgrad(theme.diverging), clims = (-m, m), xticks = (1:length(strikes), string.(Int.(strikes))), yticks = (1:length(maturities), string.(maturities)))
    for i in eachindex(maturities), j in eachindex(strikes)
        annotate!(p, j, i, text(string(round(sensitivities[i, j]; sigdigits = 2)), 7, theme.text))
    end
    return p
end

let calibration = page_module("calibration.md", "calibration", :res)
    data = Base.invokelatest(() -> (calibration.maturities, calibration.strikes, calibration.mkt.Volatility, calibration.res.ModelVolatility))
    save_themes(theme -> calibration_figure(theme, data), "calibration")
end
let exotics = page_module("exotics.md", "exotics", :∂V_∂v)
    data = Base.invokelatest(() -> (exotics.maturities, exotics.strikes, exotics.∂V_∂v))
    save_themes(theme -> exotic_figure(theme, data), "exotic")
end
