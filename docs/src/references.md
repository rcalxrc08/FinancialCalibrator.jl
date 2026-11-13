```@raw html
<div class="fc-page-head fc-refs-page">
<p class="fc-kicker"><span>Reference</span> Bibliography</p>
```

# References

```@raw html
<p class="fc-page-lede">The optimization, the implicit differentiation, the models and the Monte Carlo sensitivities
the package builds on.</p>
</div>
```

## Optimization and implicit differentiation

- K. Levenberg, *A Method for the Solution of Certain Non-Linear Problems in Least Squares*, Quarterly of Applied
  Mathematics 2(2), 164–168, 1944.
- D. W. Marquardt, *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*, Journal of the Society for
  Industrial and Applied Mathematics 11(2), 431–441, 1963. The damping scaled by the diagonal of ``J^\top J``.
- M. Blondel, Q. Berthet, M. Cuturi, R. Frostig, S. Hoyer, F. Llinares-López, F. Pedregosa and J.-P. Vert, *Efficient
  and Modular Implicit Differentiation*, Advances in Neural Information Processing Systems 35, 2022. Derivatives of the
  solution of an optimization problem from its optimality conditions.
- G. Dalle and M. Mohamed, [ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl),
  which builds the rules of [`calibrated_parameters`](@ref).

## Models and pricing

- S. L. Heston, *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency
  Options*, Review of Financial Studies 6(2), 327–343, 1993.
- H. Albrecher, P. Mayer, W. Schoutens and J. Tistaert, *The Little Heston Trap*, Wilmott Magazine, January 2007,
  83–92.
- D. Brigo and F. Mercurio, *Lognormal-Mixture Dynamics and Calibration to Market Volatility Smiles*, International
  Journal of Theoretical and Applied Finance 5(4), 427–446, 2002.
- D. Brigo, F. Mercurio and F. Rapisarda, *Smile at the Uncertainty*, Risk 17(5), 97–101, 2004. Mixtures of
  shifted lognormals, with uncertain volatilities.
- F. Fang and C. W. Oosterlee, *A Novel Pricing Method for European Options Based on Fourier-Cosine Series
  Expansions*, SIAM Journal on Scientific Computing 31(2), 826–848, 2008.
- A. L. Lewis, *A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes*, 2001,
  [SSRN 282110](https://ssrn.com/abstract=282110).
- L. B. G. Andersen, *Simple and Efficient Simulation of the Heston Stochastic Volatility Model*, Journal of
  Computational Finance 11(3), 1–42, 2008. The difficulty of simulating the variance when the Feller condition fails.

## Monte Carlo sensitivities

- F. A. Longstaff and E. S. Schwartz, *Valuing American Options by Simulation: A Simple Least-Squares Approach*, Review
  of Financial Studies 14(1), 113–147, 2001.
- P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003, chapter 7. Pathwise derivatives, and
  why they miss the discontinuities of a payoff.
