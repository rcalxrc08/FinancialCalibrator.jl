# Pricing: AnalyticMethod and the pricer hook

_Stage 02 reference._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

Files of `src/02_pricing/` (included in this order): `analytic/analytic_method.jl` (`AnalyticMethod`, the generic
closed-form `pricer`, and `LognormalMixtureKernel`, an isbits callable `(K, isCall) -> price` summing `blkprice` over a
tuple of components, and `stateless_prices`, the same prices with the numbers of the kernel as broadcast arguments of a
field-less `MixturePrice`, for the Mooncake rule on the device only, #41 group 3c), `analytic/black_scholes.jl` and `analytic/shifted_lognormal_mixture.jl` (`LogNormalMixture`, and
`ShiftedProcess{<:Union{BlackScholesProcess, LogNormalMixture}}` through `mixture_volatilities`/`mixture_component_weights`),
which define `analytic_price_type` and `analytic_kernel` for their model: a new closed form is those two methods; then
`pricer.jl` (`calibration_method`).
