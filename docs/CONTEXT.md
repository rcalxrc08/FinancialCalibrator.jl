# Documentation

## Inputs

| Load | Why |
|:--|:--|
| `make.jl`, `figures.jl`, `home.jl`, `pages.jl` | build, figures at build time, home page, page figures |
| `src/*.md` | the pages; `backends.md` holds the AD table |
| `references/docs_design.md` | the design system shared with the other packages |

## Process

Benchmarks are not timed at build time: `benchmarks.jl` is run by hand and writes `src/assets/benchmarks.toml` (method
settings at the start and at the optimum of each model, #10; optimizers with the recommended method; engines; scaling).
Measured numbers say on which machine they were taken. The "Which backend" tables of `backends.md` come from
`bench/` (its own contract).

## Outputs

`julia --project=docs docs/make.jl` builds `docs/build`; `DeployPage.yml` publishes it.
