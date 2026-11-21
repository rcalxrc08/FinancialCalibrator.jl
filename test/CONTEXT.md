# Tests

## Inputs

| Load | Why |
|:--|:--|
| `runtests.jl` | the core testset, then `logged_include` of each file |
| `context.jl` | the context layers follow the code: run it after moving, adding or renaming a file (seconds) |
| `references/testing.md` | what each file covers, durations |
| `references/workflows.md` | commands, CI (Julia 1) |

## Process

Add a test to the file of the stage it checks; a new file goes into `runtests.jl` through `logged_include`, and a new testset of `runtests.jl`, `models_methods.jl` or `gpu.jl` starts with `progress(name)`. Compare AD
backends and benchmark one fresh process per case, with the data in `const` bindings or passed as arguments (a
closure or non-const globals change what Enzyme and Zygote see).

## Outputs

`julia --project=. -e 'using Pkg; Pkg.test()'`: about 27 minutes with a GPU, 18 without (measured 2026-11-20, i7-8550U and
MX150; durations per file in `references/testing.md`).
