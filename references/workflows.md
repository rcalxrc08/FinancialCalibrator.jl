# Workflows: commands, formatting, CI, sub-environments, commits

_How to test, build the docs, format, and what CI runs._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

## Commands

```julia
julia --project=. -e 'using Pkg; Pkg.test()'     # all tests (about 1 hour: Enzyme compiles for minutes per model; 25 more minutes for the GPU part, skipped without a GPU)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=docs -e 'using Pkg; Pkg.instantiate()'; julia --project=docs docs/make.jl   # the documentation (a few minutes)
```

Formatting uses the repo's `.JuliaFormatter.toml` (not a dependency, install it in a temporary environment):

```julia
using Pkg; Pkg.activate(temp = true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")
```

Run the formatter before committing. `FinancialFFT` lives in the public LocalRegistry
(`https://github.com/rcalxrc08/LocalRegistry.git`), CI adds it with the `localregistry` input of `julia-buildpkg`.

- CI runs Julia 1; not 1.10 (Heston in the FinancialFFT example lands 2e-8 from the reference there) nor `pre` (Mooncake does not precompile on the nightly compiler). From 2026-11-14 to 2026-11-19 CI pinned 1.13.3: Enzyme aborted on Julia 1.13.4 ("unsupported tag julia.gc_alloc_ptr_offsets"); Enzyme ≥ 0.13.220 runs there (checked with 0.13.221), so the test compat asks for Enzyme 0.13.220 and CI is back on 1.
- Commit messages end with the `Co-Authored-By` line configured for the session.
- Editing files from Python on Windows: force UTF-8 (`PYTHONUTF8=1`), the sources contain non-ASCII characters.
- Python edits of Julia or Markdown files: use raw strings (`r"""..."""`), `\t` and `\n` in `\top`/`\nabla` were once
  turned into a tab and a newline in the docs.

## Sub-environments

- **Compat of the package itself in the sub-environments** (`docs/`, which load it through `[sources]` with a path): each has a `[compat]` entry for the package, so CompatHelper does not open PRs to add one (it has no option to skip a dependency). Bump these entries in the same commit as a breaking version bump, otherwise the local package no longer satisfies them and the environment fails to resolve.
