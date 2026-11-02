module FinancialCalibratorCUDAExt

using CUDA, FinancialCalibrator, FinancialMonteCarlo

# Strikes stored on the GPU are priced in CudaMode.
FinancialCalibrator.pricing_mode(::CUDA.CuArray) = FinancialMonteCarlo.CudaMode()

end
