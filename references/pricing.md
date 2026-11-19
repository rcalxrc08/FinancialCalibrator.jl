# Pricing: AnalyticMethod and the pricer hook

_Stage 02 reference._ (Layer 3 reference: stable facts, loaded only when a stage CONTEXT.md lists it.)

dropped; its cubic spline oscillated up to 2.5 bp between the nodes of the DAX rate curve), `calibrated_data.jl` (`CalibratedData`), `analytic/` (`analytic_method.jl`: `AnalyticMethod`, the
generic closed-form `pricer`, and `LognormalMixtureKernel`, an isbits callable `(K, isCall) -> price` summing `blkprice`
over a tuple of components; `black_scholes.jl` and `shifted_lognormal_mixture.jl` (`LogNormalMixture`, and `ShiftedProcess{<:Union{BlackScholesProcess, LogNormalMixture}}` through `mixture_volatilities`/`mixture_component_weights`) define `analytic_price_type` and
`analytic_kernel` for their model: a new closed form is those two methods), `pricer.jl`, `loss.jl`, `parametrization.jl`, `calibrate.jl`, `src/05_sensitivities/sensitivities.jl`.
