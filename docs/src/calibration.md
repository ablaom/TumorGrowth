# Calibration

For a basic example, see the [`CalibrationProblem`](@ref) document string below. For an exteneded workflow, see [Calibration workflows](@ref). 

## Overview

By default, calibration is performed using a gradient descent optimiser to minimise a
least-squares error on provided clinical measurements, and uses the adjoint method to
auto-differentiate solutions to the underlying ODE's, with respect to the ODE parameters,
and initial conditions to be optimised. Alternatively, Levengerg-Marquardt or Powell's dog
leg optimisation may be employed. This can be faster for the smaller models, but enjoys
fewer features.

### Gradient descent optimisation

Calibration using a gradient descent optimiser has these advantages:

- Any updater (`optimiser`) from
  [Optimisers.jl](https://fluxml.ai/Optimisers.jl/dev/) can be used.

- Fine-grained control of iteration, including live plots, is possible using
  [IterationControl.jl](https://github.com/JuliaAI/IterationControl.jl).

- Stability issues for models with a singularity at zero volume can be mitigated by
  specifying a loss `penalty`.

- Instead of minimizing a least-squares loss, one can give more weight to recent
  observations by specifying an appropriate `half_life`.

- Convergence may be faster for models with a large number of parameters (e.g., larger
  neural ODE models)

By default, optimisation is by gradient descent using Adaptive Moment Estimation and a
user-specifiable `learning_rate`.

### Levengerg-Marquardt / dog leg optimisation

The main advantage of these methods is that they are faster for all the models currently
provided, with the exception of large neural ODE models. Users can specify an initial
`trust_region_radius` to mitigate instability.

## Usage

```@docs
CalibrationProblem
```
