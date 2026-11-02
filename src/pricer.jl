# Execution mode used to price strikes stored in `StrikeVec`: serial on the host; the CUDA extension
# selects `CudaMode` for a `CuArray`.
pricing_mode(::AbstractArray) = FinancialMonteCarlo.SerialMode()
