# Draws etc/logo.png and etc/favicon.ico, in the style of the logos of FinancialToolbox.jl and FinancialFFT.jl: three
# discs in the Julia colours. A bird whose wings are a volatility surface (what is calibrated to), a smile fitted to its
# quotes (the calibration) and the level sets of a loss with the steps of Levenberg–Marquardt (the optimization, and the
# derivatives that drive it).
# Needs Luxor.jl, ForwardDiff.jl, FinancialFFT.jl, FinancialMonteCarlo.jl and FinancialToolbox.jl in the active environment: julia etc/logo.jl
using Luxor
import ForwardDiff
using FinancialFFT: CosMethod
using FinancialMonteCarlo: EuropeanOptionSmile, HestonProcess, Underlying, ZeroRate, pricer
using FinancialToolbox: blsimpv
using LinearAlgebra

const WIDTH, HEIGHT = 604, 552
const RING_RADIUS, DISC_RADIUS = 135, 117
const STROKE = 11
const BLACK = "#000000"

# Ring and disc colours of the Julia logo.
const GREEN = ("#389826", "#60ad51")
const RED = ("#cb3c33", "#d5635c")
const PURPLE = ("#9558b2", "#aa79c1")

function draw_disc(centre, colours)
    setcolor(colours[1])
    circle(centre, RING_RADIUS, :fill)
    setcolor(colours[2])
    circle(centre, DISC_RADIUS, :fill)
end

function draw_line(points)
    setcolor(BLACK)
    poly(points, :stroke, close = false)
end

# Implied volatilities of Heston, from out of the money options priced by FinancialFFT.
function heston_vols(T, log_strikes; ρ = -0.6)
    S0 = 100.0
    model = HestonProcess(0.8, 0.22, 0.0, 1.5, ρ, 0.04, Underlying(S0, 0.0))
    strikes = S0 .* exp.(log_strikes)
    calls = collect(strikes .>= S0)
    prices = pricer(model, ZeroRate(0.0), CosMethod(2^10), EuropeanOptionSmile(T, strikes, calls))
    return blsimpv.(S0, strikes, 0.0, T, prices, 0.0, calls)
end

# A bird whose wings are a volatility surface: implied volatility over strike and maturity, a mesh of faces shaded by
# their volatility (from a pale green, low, to the dark green of the ring, high), with black edges. The strikes run
# along the span of a wing, from the body to the tip, the maturities along its chord, and the volatility lifts the mesh:
# the smile bends the wing. The other wing is its mirror image.
function draw_bird()
    maturities = range(0.25, 2.0, length = 4)
    log_strikes = range(-0.3, 0.3, length = 7)
    vols = reduce(vcat, [permutedims(heston_vols(T, log_strikes; ρ = -0.2)) for T in maturities])
    low, high = extrema(vols)
    level(i, j) = (vols[i, j] - low) / (high - low)
    nT, nK = length(maturities), length(log_strikes)
    dark = Luxor.parse(Luxor.Colorant, GREEN[1])
    shade(t) = Luxor.weighted_color_mean(t, dark, Luxor.RGB(0.93, 0.97, 0.92))
    # the right wing: from the shoulder, the span goes out and up, the chord back (down), narrowing to the tip
    shoulder = (7.0, -2.0)
    function wing(i, j, side)
        u = (j - 1) / (nK - 1)
        w = (i - 1) / (nT - 1)
        chord = 46 * (1 - 0.5 * u)
        # a gull wing: the leading edge rises from the shoulder and levels off at the tip; the trailing edge sweeps back
        x = shoulder[1] + 76 * u - 14 * w * u
        y = shoulder[2] - 64 * u + 38 * u^2 + chord * w - 14 * level(i, j)
        return Point(side * x, y)
    end
    # as large as the disc allows, clear of its edge by a stroke
    magnification = 1.12
    @assert all(magnification * hypot(p.x, p.y) < DISC_RADIUS - STROKE for p in (wing(i, j, side) for i = 1:nT, j = 1:nK, side in (-1, 1)))
    scale(magnification)
    for side in (-1, 1)
        setline(0.32 * STROKE)
        for i = 1:(nT-1), j = 1:(nK-1)
            face = [wing(i, j, side), wing(i, j + 1, side), wing(i + 1, j + 1, side), wing(i + 1, j, side)]
            t = (level(i, j) + level(i, j + 1) + level(i + 1, j) + level(i + 1, j + 1)) / 4
            setcolor(shade(0.2 + 0.8 * t))
            poly(face, :fill, close = true)
            setcolor(BLACK)
            poly(face, :stroke, close = true)
        end
        setline(0.45 * STROKE)
        edge = [[wing(1, j, side) for j = 1:nK]; [wing(i, nK, side) for i = 1:nT]; [wing(nT, j, side) for j = nK:-1:1]; [wing(i, 1, side) for i = nT:-1:1]]
        setcolor(BLACK)
        poly(edge, :stroke, close = true)
    end
    # the body, the tail and the head, in the black of the strokes
    setcolor(BLACK)
    ellipse(Point(0, 16), 15, 58, :fill)
    poly([Point(-8, 38), Point(8, 38), Point(17, 64), Point(0, 56), Point(-17, 64)], :fill, close = true)
    circle(Point(0, -18), 10, :fill)
    poly([Point(-4, -26), Point(4, -26), Point(0, -38)], :fill, close = true)
end

# The calibration: quotes (discs) scattered around a smile, and the model through them (line).
function draw_fit()
    log_strikes = range(-0.4, 0.4, length = 61)
    model = heston_vols(0.25, log_strikes)
    quoted = 1:10:61
    noise = [0.02, -0.025, 0.025, -0.02, 0.025, -0.025, 0.02]
    low, high = extrema(model)
    to_point(k, v) = Point(190 * k, 48 - 100 * (v - low) / (high - low))
    setline(0.6 * STROKE)
    draw_line(to_point.(log_strikes, model))
    setcolor(BLACK)
    for (q, e) in zip(quoted, noise)
        circle(to_point(log_strikes[q], model[q] + e), STROKE, :fill)
    end
end

# The optimization: level sets of a quadratic loss, elongated and turned like the one of a calibration whose parameters
# compensate each other, and the damped Gauss–Newton steps from a far point to its minimum. The steps follow the
# gradient where the damping is large, and the curvature where it is small.
function draw_descent()
    θ = π / 5
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    A = R * Diagonal([1.0, 9.0]) * transpose(R)
    loss(x) = dot(x, A * x) / 2
    for level in (0.3, 1.2, 2.7)
        # the ellipse xᵀAx = 2 level, through the eigenvectors of A
        points = [Point((R * [sqrt(2 * level) * cos(t), sqrt(2 * level / 9) * sin(t)] .* 40)...) for t in range(0, 2π, length = 200)]
        setline(0.45 * STROKE)
        draw_line(points)
    end
    x, μ = [-2.0, 0.6], 40.0
    path = [x]
    for _ = 1:7
        g = ForwardDiff.gradient(loss, x)
        H = ForwardDiff.hessian(loss, x)
        x = x - (H + μ * Diagonal(diag(H))) \ g
        μ /= 2.5
        push!(path, x)
    end
    setline(0.6 * STROKE)
    points = [Point((40 .* p)...) for p in path]
    for (a, b) in zip(points[1:(end-1)], points[2:end])
        norm(b - a) > 2 * STROKE ? arrow(a, b, arrowheadlength = 16, arrowheadangle = π / 6, linewidth = 0.6 * STROKE) : draw_line([a, b])
    end
    circle(points[end], STROKE, :fill)
end

function draw_symbol(draw, centre)
    gsave()
    translate(centre)
    draw()
    grestore()
end

function draw_logo()
    background(1, 1, 1, 0)
    setline(STROKE)
    setlinecap("round")
    setlinejoin("round")
    top, left, right = Point(302, 145), Point(146, 415), Point(458, 415)
    for (centre, colours, draw) in ((top, GREEN, draw_bird), (left, RED, draw_fit), (right, PURPLE, draw_descent))
        draw_disc(centre, colours)
        draw_symbol(draw, centre)
    end
end

# The logo as a PNG, centred on a square of the given side.
function square_png(side)
    path = tempname() * ".png"
    Drawing(side, side, path)
    setstrokescale(true)
    scale(side / WIDTH)
    translate(0, (WIDTH - HEIGHT) / 2)
    draw_logo()
    finish()
    return read(path)
end

# An ICO file is a directory of images, here PNGs, each with its size and position in the file.
function write_ico(path, sides)
    pngs = square_png.(sides)
    open(path, "w") do io
        write(io, UInt16(0), UInt16(1), UInt16(length(pngs)))
        offset = 6 + 16 * length(pngs)
        for (side, png) in zip(sides, pngs)
            write(io, UInt8(side), UInt8(side), UInt16(0), UInt16(1), UInt16(32), UInt32(length(png)), UInt32(offset))
            offset += length(png)
        end
        foreach(png -> write(io, png), pngs)
    end
end

Drawing(WIDTH, HEIGHT, joinpath(@__DIR__, "logo.png"))
draw_logo()
finish()
write_ico(joinpath(@__DIR__, "favicon.ico"), [16, 32, 48])
