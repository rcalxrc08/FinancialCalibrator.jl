# The method the smiles of `Strikes` are priced with: the pricing method as given for strikes on the host; the CUDA
# extension moves it to CudaMode for a CuArray (FinancialMonteCarlo.with_mode), so that the smiles are priced where the
# strikes live. A method already in CudaMode moves host strikes to the device by itself.
calibration_method(method::FinancialMonteCarlo.AbstractMethod, ::AbstractArray) = method
