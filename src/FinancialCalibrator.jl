module FinancialCalibrator

using Dates, Random, LinearAlgebra, DelimitedFiles
using Adapt, Dierckx, FFTW, FinancialFFT, FinancialMonteCarlo, FinancialToolbox, JSON, Optim
import FinancialMonteCarlo: pricer

# Directory containing the bundled example market data.
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data", "json"))

include("structs.jl")
include("retriever_calibration.jl")
include("adapter_calibration.jl")
include("calibrated_data.jl")
include("shifted_lognormal_mixture_pricer.jl")
include("pricer.jl")
include("loss.jl")
include("calibrate.jl")

export CalibrationID, IOFileInputStrategy, retriever, adapter, calibrate
export AnalyticMethod, CalibratedData, MarketData, RawMarketData

end # module
