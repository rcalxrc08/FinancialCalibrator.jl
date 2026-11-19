# Documentation design

_The Documenter site: design system, figures, home page, benchmarks._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

The documentation is Documenter.jl, in the style of FinancialFFT.jl and of the website (the shared assets are
`docs/src/assets/style.css`, with the "Dark Quant" palette of rcalxrc08.github.io, `dark.js`, which makes the dark theme
the default for a reader who has not picked one, and `figures.css`; #26), hero on `index.md`, figures computed at build
time by `docs/figures.jl` in a light and a dark variant (the dark one in the Dark Quant colours), from the
`@example` blocks of the pages run in a module of their own). The whole site has one design system, `docs/src/assets/site.css` (`fc-` classes: page
header `fc-page-head` with kicker, lede, `fc-meta` chips and an `fc-toc` strip of section links, numbered `h2`, figure
cards, maths panels, tile rows `fc-stats`, cards `fc-card`, docstring and reference cards, previous/next cards); the
sidebar groups the pages into Guides and Reference with short names (`make.jl`). The guides get their key numbers from
hidden `@example` blocks calling `Main.DocumentationFigures` (`calibration_tiles`, `closed_form_tiles`, `exotic_tiles`,
`engine_table`) and their extra figures (singular values, ∂p*/∂v, closed form against Black–Scholes) from
`docs/pages.jl`; `page_module` skips those callback blocks. The "Similar packages" section (ORE, DAL, finmath-lib, QuantLibAAD, stochastic-rs, Hedgehog.jl; surveyed 2026-11-15, features only, no quality or speed claims) is the same text at the end of `README.md` and before "The ecosystem" in `index.md`: edit both together. The home page (`index.md`, sidebar name "Home") is a landing page: raw HTML
sections styled by `docs/src/assets/home.css` (scoped with `.content:has(.fc-hero)`, `fc-` classes; the hero opens a
`<div>` in one `@raw html` block and closes it in the next, around the Markdown `#` title), figures and numbers from
`docs/home.jl` (included by `figures.jl`): a Heston calibration to DAX drawn as a perspective SVG (`surface_svg`, hand
written, painter's algorithm), smiles, residuals, ∂p*/∂v heatmaps, a loss landscape (shown full width one at a time, behind tabs made of radio
buttons and CSS, no script), and the benchmark charts. The
stat tiles and the engines table are `HTML` returned by hidden `@example` blocks calling `Main.DocumentationFigures`.
Benchmarks are **not** timed at build time: `docs/benchmarks.jl` is run by hand on the laptop (i7-8550U, MX150) and
writes `docs/src/assets/benchmarks.toml`, which the build reads; the AD chart is the table of `backends.md`
(`AD_TABLE` in `home.jl`). The visualisation palette is blue, teal, amber, cyan, green (`VIZ` in `home.jl`,
`THEMES` in `figures.jl`), darker steps in the light theme. `figures.css` hides the other theme's figure with
`.content img[...]` selectors (the themes' `.content figure img` rule otherwise showed both). Local preview with
pretty URLs: `CI=true julia --project=docs docs/make.jl`, then serve `docs/build` over HTTP. `.github/workflows/DeployPage.yml` builds it and publishes it
with `actions/deploy-pages` (the Pages source of the repository must be "GitHub Actions"); the Franklin `page/` site was
removed. The logo and favicon are drawn by `etc/logo.jl` (Luxor, run in a temporary environment): a bird whose wings are a
Heston vol surface, a smile fitted to quotes, loss level sets with the Levenberg–Marquardt path; `docs/make.jl` copies them
into the assets.
