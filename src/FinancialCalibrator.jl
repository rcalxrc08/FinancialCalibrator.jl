module FinancialCalibrator

using Dates, LinearAlgebra, Downloads
using ADTypes, Adapt, Dierckx, FinancialMonteCarlo, FinancialToolbox, JSON, Optim
# ADTypes and DifferentiationInterface are interfaces, and dependencies of Optim (through NLSolversBase) anyway, and
# ImplicitDifferentiation builds the rules of calibrated_parameters on them: the packages that differentiate are weak
# dependencies, see backends.jl and ext/.
import DifferentiationInterface as DI
using ImplicitDifferentiation: ImplicitFunction, MatrixRepresentation, DirectLinearSolver
import FinancialMonteCarlo: pricer

# Directory containing the bundled example market data.
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data", "json"))

include("structs.jl")
include("retriever_calibration.jl")
include("cboe.jl")
include("adapter_calibration.jl")
include("calibrated_data.jl")
include("shifted_lognormal_mixture_pricer.jl")
include("pricer.jl")
include("loss.jl")
include("parametrization.jl")
include("backends.jl")
include("calibrate.jl")
include("sensitivities.jl")

export CalibrationID, IOFileInputStrategy, CboeInputStrategy, CboeWebInputStrategy, cboe_snapshot, retriever, adapter, calibrate, calibrated_parameters
export AnalyticMethod, CalibratedData, MarketData, RawMarketData, Parametrization, vol_errors

end # module
