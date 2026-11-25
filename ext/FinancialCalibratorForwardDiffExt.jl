# The value of a ForwardDiff dual number without its partials, nested duals included: the loss prices it again to tell
# parameters that cannot be priced from a failure of the derivatives only (#42). Replace with
# FinancialToolbox.primal_value once it is public (rcalxrc08/FinancialToolbox.jl#81).
module FinancialCalibratorForwardDiffExt

import FinancialCalibrator
using ForwardDiff: Dual, value

FinancialCalibrator.primal_value(x::Dual) = FinancialCalibrator.primal_value(value(x))

end
