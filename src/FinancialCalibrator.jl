module FinancialCalibrator

using Dates, LinearAlgebra, Downloads
using ADTypes, Adapt, ChainRulesCore, FinancialMonteCarlo, FinancialToolbox, JSON, Optim
# ADTypes and DifferentiationInterface are interfaces, and dependencies of Optim (through NLSolversBase) anyway, and
# ImplicitDifferentiation builds the rules of calibrated_parameters on them: the packages that differentiate are weak
# dependencies, see backends.jl and ext/.
import DifferentiationInterface as DI
using ImplicitDifferentiation: ImplicitFunction, MatrixRepresentation, DirectLinearSolver
import FinancialMonteCarlo: pricer
using FinancialMonteCarlo: BlackScholesProcess, LogNormalMixture, ShiftedProcess

# Directory containing the bundled example market data.
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data", "json"))

# The stages of a calibration, in order (each folder has a CONTEXT.md: its inputs, what it does, what it provides)
# 01: quotes in, MarketData and the Underlying out
include("01_market_data/structs.jl")
include("01_market_data/retriever_calibration.jl")
include("01_market_data/cboe.jl")
include("01_market_data/adapter_calibration.jl")
# 02: a model and a method price the smiles (closed forms here, any FinancialMonteCarlo method otherwise)
include("02_pricing/analytic/analytic_method.jl")
include("02_pricing/analytic/black_scholes.jl")
include("02_pricing/analytic/shifted_lognormal_mixture.jl")
include("02_pricing/pricer.jl")
# 03: the volatility errors, their loss, the variables of the search, the default AD backend
include("03_loss/loss.jl")
include("03_loss/parametrization.jl")
include("03_loss/backends.jl")
# 04: the optimizer and the Levenberg-Marquardt polish
include("04_calibration/calibrated_data.jl")
include("04_calibration/calibrate.jl")
# 05: the calibrated parameters as a differentiable function of the quotes (implicit function theorem)
include("05_sensitivities/sensitivities.jl")

export CalibrationID, IOFileInputStrategy, CboeInputStrategy, CboeWebInputStrategy, cboe_snapshot, retriever, adapter, calibrate, calibrated_parameters
export AnalyticMethod, CalibratedData, MarketData, RawMarketData, Parametrization, default_parametrization, vol_errors

end # module
