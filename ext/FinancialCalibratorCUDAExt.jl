module FinancialCalibratorCUDAExt

using CUDA, FinancialCalibrator, FinancialMonteCarlo

# Strikes stored on the GPU are priced in CudaMode.
FinancialCalibrator.calibration_method(method::FinancialMonteCarlo.AbstractMethod, ::CUDA.CuArray) = FinancialMonteCarlo.with_mode(method, FinancialMonteCarlo.CudaMode())

end
