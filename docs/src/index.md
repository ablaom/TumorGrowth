# Overview

TumorGrowth.jl provides the following models (ODE solvers) for tumor growth:

| model                           | description                             | parameters, `p`       | analytic? | ODE                                     |
|:--------------------------------|:----------------------------------------|:----------------------|:----------|:----------------------------------------|
| [`bertalanffy`](@ref)           | generalized Bertalanffy (GB)            | `(; v0, v∞, ω, λ)`    | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`bertalanffy_numerical`](@ref) | generalized Bertalanffy (testing only)  | `(; v0, v∞, ω, λ)`    | no        | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`bertalanffy2`](@ref)          | 2D extension of generalized Bertalanffy | `(; v0, v∞, ω, λ, γ)` | no        | [`TumorGrowth.bertalanffy2_ode!`](@ref) |
| [`gompertz`](@ref)              | Gompertz (GB, `λ=0`)                    | `(; v0, v∞, ω)`       | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`logistic`](@ref)              | logistic/Verhulst (GB, `λ=-1`)          | `(; v0, v∞, ω)`       | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`classical_bertalanffy`](@ref) | classical Bertalanffy (GB, `λ=1/3`)     | `(; v0, v∞, ω)`       | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`neural2(rng, network)`](@ref) | neural2 ODE with Lux.jl `network`       | `(; v0, v∞, θ)`       | no        | [`TumorGrowth.neural_ode`](@ref)        |

The models predict a sequence of lesion volumes, given times and parameters:

```@example overview
using TumorGrowth

times = times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
p = (v0=0.0002261, v∞=2.792e-5,  ω=0.05731) # `v0` is the initial volume
gompertz(times, p)
```

The underlying ODEs are solved under the hood, if an analytic solution is not known.

TumorGrowth.jl also provides a [`CalibrationProblem`](@ref) tool to calibrate model
parameters, given a history of measurements, and a [`compare`](@ref) tool to compare models
on a holdout set.
