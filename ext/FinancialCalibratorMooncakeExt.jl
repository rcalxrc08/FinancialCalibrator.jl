# A Mooncake rule for the solution of the implicit function of calibrated_parameters (FinancialCalibrator.implicit_variables). ImplicitDifferentiation gives a
# ChainRules rrule for it, which takes a RuleConfig to choose the backends of the conditions; Mooncake's @from_rrule
# calls rules without one, and converts tangents well only for simple types (the implicit function is a struct of
# closures). calibrated_parameters always passes its backends, so the config is never used: a config without modes
# reaches the rrule, and the rule below moves the tangents of the quotes between Mooncake and ChainRules.
module FinancialCalibratorMooncakeExt

using FinancialCalibrator
using ChainRulesCore: ChainRulesCore, RuleConfig, NoForwardsMode, NoReverseMode, NoTangent, unthunk
using ImplicitDifferentiation: ImplicitFunction
using Mooncake: Mooncake, CoDual, NoRData, primal, tangent, zero_rdata, zero_fcodual, build_rrule, @is_primitive, @zero_derivative, DefaultCtx, ReverseMode

# The setup of the implicit function does not depend on the quotes being differentiated (they enter through
# implicit_variables), so it has no derivative; without this, Mooncake would differentiate it, and fail on the copies
# of the quotes from the GPU. Its result holds the arguments (the model, the market data), but no tangent flows into it:
# the rule of implicit_variables gives the implicit function no cotangent.
@zero_derivative DefaultCtx Tuple{typeof(FinancialCalibrator.implicit_calibration), Vararg}

struct NoModesConfig <: RuleConfig{Union{NoForwardsMode, NoReverseMode}} end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(FinancialCalibrator.implicit_variables), ImplicitFunction, AbstractMatrix{<:Base.IEEEFloat}}

function Mooncake.rrule!!(::CoDual{typeof(FinancialCalibrator.implicit_variables)}, implicit::CoDual{<:ImplicitFunction}, vols::CoDual{<:AbstractMatrix{<:Base.IEEEFloat}})
    (y, z), pullback = ChainRulesCore.rrule(NoModesConfig(), primal(implicit), primal(vols))
    # the tangent of the variables, an array: Mooncake accumulates into it before calling the pullback
    ȳ = zero(y)
    function implicit_variables_pullback(::NoRData)
        dvols = unthunk(pullback((ȳ, NoTangent()))[2])
        tangent(vols) .+= dvols
        # the implicit function holds numbers (the data of the calibration): its rdata is not NoRData, and it gets no
        # cotangent, as in the rrule of ImplicitDifferentiation
        return NoRData(), zero_rdata(primal(implicit)), NoRData()
    end
    return CoDual(y, ȳ), implicit_variables_pullback
end

# The loss: Mooncake rejects the try/catch of loss in reverse mode (#40), so this rule runs the try itself around the
# rule Mooncake derives for the loss without it (built once per signature: build_rrule caches it and returns a copy).
# Where the forward pass throws a numerical failure the loss is Inf with a zero gradient, as with ForwardDiff and Enzyme,
# and as the ChainRules rrule gives Zygote (which Mooncake.@from_rrule cannot call: it needs a RuleConfig).
@is_primitive DefaultCtx ReverseMode Tuple{typeof(FinancialCalibrator.loss), Type, AbstractVector{<:Real}, Vararg}

function Mooncake.rrule!!(::CoDual{typeof(FinancialCalibrator.loss)}, R::CoDual, args::Vararg{CoDual, N}) where {N}
    loss_value = FinancialCalibrator.loss_value
    rule = build_rrule(loss_value, primal(R), map(primal, args)...)
    try
        value, value_pullback = rule(zero_fcodual(loss_value), R, args...)
        loss_pullback(dy) = (NoRData(), Base.tail(value_pullback(dy))...)
        return value, loss_pullback
    catch e
        value = FinancialCalibrator.failed_loss(primal(R), primal(first(args)), e)
        failure_pullback(_) = (NoRData(), NoRData(), map(a -> zero_rdata(primal(a)), args)...)
        return zero_fcodual(value), failure_pullback
    end
end

end # module
