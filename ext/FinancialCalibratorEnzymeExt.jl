# Enzyme for calibrated_parameters. ImplicitDifferentiation has Enzyme rules for an ImplicitFunction annotated `Const`:
# the implicit function is built inside the differentiated code (implicit_calibration), so Enzyme would annotate it as
# active (`Duplicated`) and no rule would match. The setup does not depend on the quotes being differentiated (they enter
# through implicit_variables), so it is inactive: its result is a constant, and ImplicitDifferentiation's rules apply.
# Those rules also need the solver to be inferred (they check the type of the shadow against the type of the solution):
# optimal_variables asserts the type of its result.
module FinancialCalibratorEnzymeExt

using FinancialCalibrator
using EnzymeCore: EnzymeRules

EnzymeRules.inactive(::typeof(FinancialCalibrator.implicit_calibration), args...) = nothing

end # module
