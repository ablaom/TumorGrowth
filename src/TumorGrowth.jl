module TumorGrowth

import DifferentialEquations as DE
import SciMLSensitivity as Sens
import CSV
import Tables
using Optimisers
using IterationControl
using Random
import Functors
import Lux
using UnPack
using ComponentArrays
using Statistics

include("plots.jl")
include("patient_data.jl")
include("tools.jl")
include("functor.jl")
include("odes.jl")
include("solutions.jl")
include("pretty.jl")
include("optimisers.jl")
include("calibration.jl")
include("model_comparison.jl")

export patient_data,
    flat_patient_data,
    bertalanffy,
    bertalanffy_numerical,
    bertalanffy2,
    gompertz,
    logistic,
    classical_bertalanffy,
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
    errors

# for julia < 1.9
if !isdefined(Base, :get_extension)
  include("../ext/PlotsExt.jl")
end

end # module Bertalanffy
