# Quality checks of Aqua.jl: ambiguities, piracy, unbound type parameters, stale dependencies, compat entries.
using Aqua
@testset "Aqua" begin
    Aqua.test_all(FinancialCalibrator)
end
