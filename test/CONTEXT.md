# Tests

## Inputs

| Load | Why |
|:--|:--|
| `runtests.jl` | the core testset, then `logged_include` of each file |
| `context.jl` | the context layers follow the code: run it after moving, adding or renaming a file (seconds) |
| `references/testing.md` | what each file covers, durations |
| `references/workflows.md` | commands, CI (Julia 1) |

## Process

Add a test to the file of the stage it checks; a new file goes into `runtests.jl` through `logged_include`. Compare AD
backends and benchmark one fresh process per case, with the data in `const` bindings or passed as arguments (a
closure or non-const globals change what Enzyme and Zygote see).

## Outputs

`julia --project=. -e 'using Pkg; Pkg.test()'`: about an hour (Enzyme compiles for minutes per model), 25 more minutes
on a GPU.
