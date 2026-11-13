"""
    Parametrization(to_params = identity, from_params = identity)

The variables a calibration searches over. `to_params(x)` gives the parameters of the model, in the order of
`FinancialMonteCarlo.get_parameters`, from the variables `x`; `from_params(p)` is its inverse, which gives the starting
point from the parameters of the model to calibrate. `Parametrization()` searches over the parameters themselves; the
default of [`calibrate`](@ref) and [`calibrated_parameters`](@ref) is [`default_parametrization`](@ref) of the model.

A change of variables keeps every step of the search on a valid model (a positive volatility, a correlation in
``(-1, 1)``), and fixes the parameters that the quotes cannot determine. For Heston, whose parameters are
``(\\sigma, \\sigma_0, \\lambda, \\kappa, \\rho, \\theta)``, with no market price of volatility risk:

```julia
heston_variables = Parametrization(x -> [exp(x[1]), exp(x[2]), zero(x[1]), exp(x[3]), tanh(x[4]), exp(x[5])],
                                   p -> [log(p[1]), log(p[2]), log(p[4]), atanh(p[5]), log(p[6])])
```

`to_params` must be generic in the number type of `x`: the calibration differentiates through it.
"""
struct Parametrization{F, G}
    to_params::F
    from_params::G
end
Parametrization() = Parametrization(identity, identity)

@doc raw"""
    default_parametrization(model)

The [`Parametrization`](@ref) a calibration of `model` uses by default: unconstrained variables, so that every point the
optimizer tries is a valid model, constraints between the parameters included. The parameters keep the order of
`FinancialMonteCarlo.get_parameters`:

| Model | Parameters | Variables |
|:--|:--|:--|
| `BlackScholesProcess` | ``\sigma`` | ``\sigma = e^{x}`` |
| `MertonProcess` | ``\sigma, \lambda, \mu_J, \sigma_J`` | ``e^{x_1}, e^{x_2}, x_3, e^{x_4}`` |
| `KouProcess` | ``\sigma, \lambda, p, \lambda_+, \lambda_-`` | ``e^{x_1}, e^{x_2}, 1/(1 + e^{-x_3}), 1 + e^{x_4}, e^{x_5}`` |
| `VarianceGammaProcess` | ``\sigma, \theta, \kappa`` | ``e^{x_1}``, ``1/\kappa - \sigma^2/2 - e^{x_2}``, ``e^{x_3}`` |
| `NormalInverseGaussianProcess` | ``\sigma, \theta, \kappa`` | ``e^{x_1}``, ``(1/\kappa - \sigma^2)/2 - e^{x_2}``, ``e^{x_3}`` |
| `HestonProcess` | ``\sigma, \sigma_0, \lambda, \kappa, \rho, \theta`` | ``e^{x_1}, e^{x_2}, 0, e^{x_3}, \tanh x_4, e^{x_5}`` |
| `ShiftedLogNormalMixture` | ``\eta_1 \dots \eta_n, \lambda_1 \dots \lambda_{n-1}, \alpha`` | ``\eta_i = e^{x_i}``, ``\lambda_i = e^{y_i} / (1 + \sum_j e^{y_j})``, ``\alpha`` |
| `LogNormalMixture` | ``\eta_1 \dots \eta_n, \lambda_1 \dots \lambda_{n-1}`` | as the shifted mixture, without ``\alpha`` |

The ``\theta`` of Variance Gamma and NIG is bounded above by the other two parameters (the martingale correction must
exist), and the weights of a mixture sum to less than one. Heston's market price of volatility risk ``\lambda`` is
fixed at zero: the quotes cannot separate it from ``\kappa`` (the Hessian of the loss is singular, and so is the implicit
derivative of [`calibrated_parameters`](@ref)), so it is not calibrated. Any other model searches over its parameters
(`Parametrization()`).

The starting point must satisfy the constraints strictly: a parameter on the boundary has no finite variable.
"""
default_parametrization(::FinancialMonteCarlo.BaseProcess) = Parametrization()
default_parametrization(::FinancialMonteCarlo.BlackScholesProcess) = Parametrization(x -> exp.(x), p -> log.(p))
default_parametrization(::FinancialMonteCarlo.MertonProcess) = Parametrization(x -> [exp(x[1]), exp(x[2]), x[3], exp(x[4])], p -> [log(p[1]), log(p[2]), p[3], log(p[4])])
default_parametrization(::FinancialMonteCarlo.KouProcess) = Parametrization(x -> [exp(x[1]), exp(x[2]), inv(1 + exp(-x[3])), 1 + exp(x[4]), exp(x[5])], p -> [log(p[1]), log(p[2]), log(p[3] / (1 - p[3])), log(p[4] - 1), log(p[5])])

# Variance Gamma and NIG: θ below the bound that σ and κ set, 1/κ - σ²/2 (VG) or (1/κ - σ²)/2 (NIG)
vg_theta_bound(σ, κ) = inv(κ) - σ^2 / 2
nig_theta_bound(σ, κ) = (inv(κ) - σ^2) / 2
bounded_theta(bound) = Parametrization(x -> [exp(x[1]), bound(exp(x[1]), exp(x[3])) - exp(x[2]), exp(x[3])], p -> [log(p[1]), log(bound(p[1], p[3]) - p[2]), log(p[3])])
default_parametrization(::FinancialMonteCarlo.VarianceGammaProcess) = bounded_theta(vg_theta_bound)
default_parametrization(::FinancialMonteCarlo.NormalInverseGaussianProcess) = bounded_theta(nig_theta_bound)

default_parametrization(::FinancialMonteCarlo.HestonProcess) = Parametrization(x -> [exp(x[1]), exp(x[2]), zero(x[1]), exp(x[3]), tanh(x[4]), exp(x[5])], p -> [log(p[1]), log(p[2]), log(p[4]), atanh(p[5]), log(p[6])])

# Mixtures: n positive volatilities, then n - 1 weights that are positive and sum to less than one (the last weight is
# what is left), as a softmax with the remainder as one more class; the shifted mixture ends with its free shift α.
mixture_weights(y) = (w = exp.(y); w ./ (1 + sum(w)))
mixture_variables(λ) = log.(λ ./ (1 - sum(λ)))
default_parametrization(::FinancialMonteCarlo.LogNormalMixture) = Parametrization(x -> (n = (length(x) + 1) ÷ 2; vcat(exp.(x[1:n]), mixture_weights(x[(n+1):end]))), p -> (n = (length(p) + 1) ÷ 2; vcat(log.(p[1:n]), mixture_variables(p[(n+1):end]))))
default_parametrization(::FinancialMonteCarlo.ShiftedLogNormalMixture) = Parametrization(x -> (n = length(x) ÷ 2; vcat(exp.(x[1:n]), mixture_weights(x[(n+1):(end-1)]), x[end:end])), p -> (n = length(p) ÷ 2; vcat(log.(p[1:n]), mixture_variables(p[(n+1):(end-1)]), p[end:end])))
