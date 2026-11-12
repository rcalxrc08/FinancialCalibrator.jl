"""
    Parametrization(to_params = identity, from_params = identity)

The variables a calibration searches over. `to_params(x)` gives the parameters of the model, in the order of
`FinancialMonteCarlo.get_parameters`, from the variables `x`; `from_params(p)` is its inverse, which gives the starting
point from the parameters of the model to calibrate. The default searches over the parameters themselves.

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
