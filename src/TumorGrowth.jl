"""

TumorGrowth.jl provides the following models for tumor growth:

| model                           | description                             | parameters, `p`       | analytic? |
|:--------------------------------|:----------------------------------------|:----------------------|:----------|
| [`bertalanffy`](@ref)           | General Bertalanffy (GB)                | `(; v0, v∞, ω, λ)`    | yes       |
| [`bertalanffy_numerical`](@ref) | General Bertalanffy (testing only)      | `(; v0, v∞, ω, λ)`    | no        |
| [`bertalanffy2`](@ref)          | 2D extension of General Bertalanffy     | `(; v0, v∞, ω, λ, γ)` | no        |
| [`gompertz`](@ref)              | classical Gompertz (GB, `λ=0`)          | `(; v0, v∞, ω)`       | yes       |
| [`logistic`](@ref)              | classical Logistic/Verhulst (GB, `λ=-1`) | `(; v0, v∞, ω)`      | yes       |
| [`classical_bertalanffy`](@ref) | classical Bertalanffy (GB, `λ=1/3`)     | `(; v0, v∞, ω)`       | yes       |
| [`exponential`](@ref)           | exponential decay or growth             | `(; v0, ω)`           | yes       |
| [`neural(rng, network)`](@ref)  | 1D neural ODE with Lux.jl `network`     | `(; v0, v∞, θ)`       | no        |
| [`neural2(rng, network)`](@ref) | 2D neural ODE with Lux.jl `network`     | `(; v0, v∞, θ)`       | no        |

Here a *model* is a callable object, that outputs a sequence of lesion volumes, given
times, by solving a related ordinary differential equation with parameters (`p` below):

```julia
using TumorGrowth

times = times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
p = (v0=0.0002261, v∞=2.792e-5,  ω=0.05731) # `v0` is the initial volume
volumes = gompertz(times, p)
6-element Vector{Float64}:
 0.0002261
 0.0001240760197801191
 6.473115210101774e-5
 4.751268597529182e-5
 3.9074807723757934e-5
 3.496675045077041e-5
```

In every model, `v0` is the initial volume, so that `volumes[1] == v0`.

In the case analytic solutions to the underlying ODEs are not known, optional keyword
arguments for the [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
solver can be passed to the model call.

TumorGrowth.jl also provides a [`CalibrationProblem`](@ref) tool to calibrate model
parameters, and a [`compare`](@ref) tool to compare models on a holdout set.

"""
module TumorGrowth

import DifferentialEquations as DE
import SciMLSensitivity as Sens  # used also for Zygote
import LeastSquaresOptim as LSO
import LeastSquaresOptim: LevenbergMarquardt, Dogleg
using Suppressor
import CSV
import Tables
using Optimisers
using IterationControl
import Functors
import Lux
using UnPack
using ComponentArrays

using Statistics
using Random


include("plots.jl")
include("patient_data.jl")
include("tools.jl")
include("functor.jl")
include("odes.jl")
include("optimisers.jl")
include("calibration.jl")
include("compare.jl")
include("api.jl")
include("models/shared.jl")
include("models/bertalanffy.jl")
include("models/bertalanffy2.jl")
include("models/bertalanffy_numerical.jl")
include("models/classical_bertalanffy.jl")
include("models/gompertz.jl")
include("models/logistic.jl")
include("models/exponential.jl")
include("models/neural.jl")
include("models/neural2.jl")
include("pretty.jl")

export patient_data,
    flat_patient_data,
    bertalanffy,
    bertalanffy_numerical,
    bertalanffy2,
    gompertz,
    logistic,
    classical_bertalanffy,
    exponential,
    neural2,
    neural,
    CalibrationProblem,
    compare,
    Neural,
    Neural2,
    ModelComparison,
    solve!,
    loss,
    guess_parameters,
    pretty,
    solution,
    parameters,
    errors,
    Dogleg,
    LevenbergMarquardt,
    LeastSquaresOptim,
    Optimisers

# for julia < 1.9
if !isdefined(Base, :get_extension)
  include("../ext/PlotsExt.jl")
end

end # module Bertalanffy
