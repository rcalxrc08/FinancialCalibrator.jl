# Enzyme rules for calibrated_variables, the implicit function of calibrated_parameters built and called. Enzyme cannot
# trace the construction of the implicit function (implicit_calibration): even marked inactive, the struct it returns
# (the market data, the cache, on the GPU device pointers) goes on Enzyme's tape, which fails (a tape type mismatch on
# the host when the solver was not inferred, an LLVM assertion with GPU quotes, #21). The rules below build it as plain
# Julia code and hand it, annotated `Const`, to the Enzyme rules of ImplicitDifferentiation, which return the variables
# and a byproduct: only the variables are kept. The other arguments (model, market data, backends, ...) get no
# derivative, as in the ChainRules rule.
module FinancialCalibratorEnzymeExt

using FinancialCalibrator: calibrated_variables, implicit_calibration
using EnzymeCore: Active, Annotation, BatchDuplicated, Const, Duplicated
using EnzymeCore.EnzymeRules: EnzymeRules, FwdConfig, RevConfig, AugmentedReturn, needs_primal, needs_shadow

# the variables of (y, z) and of its shadows, for each return configuration of a forward rule
variables_return(::FwdConfig{true, true, 1}, r) = Duplicated(first(r.val), first(r.dval))
variables_return(::FwdConfig{true, true}, r) = BatchDuplicated(first(r.val), map(first, r.dval))
variables_return(::FwdConfig{false, true, 1}, r) = first(r)
variables_return(::FwdConfig{false, true}, r) = map(first, r)
variables_return(::FwdConfig{true, false}, r) = first(r)
variables_return(::FwdConfig{false, false}, r) = nothing

function EnzymeRules.forward(config::FwdConfig, ::Const{typeof(calibrated_variables)}, RT::Type{<:Annotation}, vols::Annotation, setup::Vararg{Annotation, N}) where {N}
    implicit = implicit_calibration(map(a -> a.val, setup)...)
    return variables_return(config, EnzymeRules.forward(config, Const(implicit), RT, vols))
end

variables_shadow(::RevConfig{<:Any, true, 1}, shadow) = first(shadow)
variables_shadow(::RevConfig{<:Any, true}, shadow) = map(first, shadow)
variables_shadow(::RevConfig{<:Any, false}, shadow) = nothing

function EnzymeRules.augmented_primal(config::RevConfig, ::Const{typeof(calibrated_variables)}, RT::Type{<:Annotation}, vols::Annotation, setup::Vararg{Annotation, N}) where {N}
    implicit = implicit_calibration(map(a -> a.val, setup)...)
    # the shadows of the variables are those of ImplicitDifferentiation's rule: its reverse pass reads what Enzyme
    # accumulates into them
    r = EnzymeRules.augmented_primal(config, Const(implicit), RT, vols)
    primal = needs_primal(config) ? first(r.primal) : nothing
    return AugmentedReturn(primal, variables_shadow(config, r.shadow), (implicit, r.tape))
end

function EnzymeRules.reverse(config::RevConfig, ::Const{typeof(calibrated_variables)}, RT::Type{<:Annotation}, tape, vols::Annotation, setup::Vararg{Annotation, N}) where {N}
    implicit, implicit_tape = tape
    EnzymeRules.reverse(config, Const(implicit), RT, implicit_tape, vols)
    # no derivative for the setup; an active number (min_vega) needs a zero
    return (nothing, map(a -> a isa Active ? zero(a.val) : nothing, setup)...)
end

end # module
