# # Calibration workflows

# This demonstration is also available in
# [notebook](https://github.com/ablaom/TumorGrowth.jl/tree/dev/docs/src/examples/03_calibration)
# form, and has been tested in the Julia package environment specified by
# [these](https://github.com/ablaom/TumorGrowth.jl/tree/dev/docs/src/examples/03_calibration)
# Project.toml and Manifest.toml files.

using Pkg #hide
dir = @__DIR__ #hide
Pkg.activate(dir) #hide
Pkg.instantiate() #hide

using TumorGrowth
using Statistics
using Plots
using IterationControl

Plots.scalefontsizes() # reset font sizes
Plots.scalefontsizes(0.85)
nothing #src

# ## Data ingestion

# Get the records which have a least 6 measurements:

records = patient_data();
records6 = filter(records) do record
    record.readings >= 6
end;


# ## Helpers

# Wrapper to only apply control every 100 steps:

sometimes(control) = IterationControl.skip(control, predicate=100)

# Wrapper to only apply control after first 30 steps:

warmup(control) = IterationControl.Warmup(control, 30)


# ## Patient A - a volume that is mostly decreasiing

record = records6[2]

#-

times = record.T_weeks
volumes = record.Lesion_normvol;

# We'll try calibrating the General Bertalanffy model, `bertalanffy`, with fixed
# parameter `λ=1/5`:

problem = CalibrationProblem(
    times,
    volumes,
    bertalanffy;
    frozen=(; λ=1/5),
    learning_rate=0.001,
    half_life=21, # place greater weight on recent measurements
)

# The controls in the `solve!` call below have the following interpretations:

# - `Step(1)`: compute 1 iteration at a time
# - `InvalidValue()`: to catch parameters going out of bounds
# - `NumberLimit(6000)`: stop after 6000 steps
# - `GL() |> warmup`:  stop using Prechelt's GL criterion after the warm-up period
# - `NumberSinceBest(10) |> warmup`:  stop when it's 10 steps since the best so far
# - `Callback(prob-> (plot(prob); gui())) |> sometimes`: periodically plot the problem

# Some other possible controls are:

# - `TimeLimit(1/60)`: stop after 1 minute
# - `WithLossDo()`: log to `Info` the current loss

# See
# [IterationControl.jl](https://github.com/JuliaAI/IterationControl.jl?tab=readme-ov-file#controls-provided)
# for a complete list.

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
    GL() |> warmup,
    NumberSinceBest(10)  |> warmup,
    Callback(prob-> (plot(prob); gui())) |> sometimes,
)

#-

p = solution(problem)
extended_times = vcat(times, [40.0, 47.0])
bertalanffy(extended_times, p)

#-

plot(problem, title="Patient A, λ=1/5 fixed", color=:black)

#-

savefig(joinpath(dir, "patientA.png"))

#-


# ## Patient B - relapse following initial improvement

record = records6[10]

times = record.T_weeks
volumes = record.Lesion_normvol;

# We'll first try the earlier simple model, but we won't freeze `λ`. Also, we won't
# specify a `half_life`, giving all the data equal weight.

problem = CalibrationProblem(
    times,
    volumes,
    bertalanffy;
    learning_rate=0.001,
)

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
)
plot(problem, label="bertalanffy")

#-

# Let's try the 2D generalization of the General Bertalanffy model:

problem = CalibrationProblem(
    times,
    volumes,
    bertalanffy2;
    learning_rate=0.001,
)

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
)
plot!(problem, label="bertalanffy2")

# Here's how we can inspect the final parameters:

solution(problem)

# Or we can do:

solution(problem) |> pretty

# And finally, we'll try a 2D neural ODE model, with fixed volume scale `v∞`.

using Lux, Random

# *Note well* the zero-initialization of weights in first layer:

network2 = Chain(
    Dense(2, 2, Lux.tanh, init_weight=Lux.zeros64),
    Dense(2, 2),
)

# Notice this network has a total of 12 parameters. To that we'll be adding the initial
# value `u0` of the latent variable. So this is a model with relatively high complexity
# for this problem.

n2 = neural2(Xoshiro(123), network2) # `Xoshiro` is a random number generator

# Note the reduced learning rate.

v∞ = mean(volumes)

problem = CalibrationProblem(
    times,
    volumes,
    n2;
    frozen = (; v∞),
    learning_rate=0.001,
)

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
)
plot!(
    problem,
    title = "Model comparison for Patient B",
    label = "neural2",
    legend=:inside,
)

#-

savefig(joinpath(dir, "patientB.png"))

# For a more principled comparison, we compare the models on a holdout set. We'll
# additionally throw in 1D neural ODE model.

network1 = Chain(
    Dense(1, 3, Lux.tanh, init_weight=Lux.zeros64),
    Dense(3, 1),
)

n1 = neural(Xoshiro(123), network1)

models = [bertalanffy, bertalanffy2, n1, n2]
calibration_options = [
    (frozen = (; λ=1/5), learning_rate=0.001, half_life=21), # bertalanffy
    (frozen = (; λ=1/5), learning_rate=0.001, half_life=21), # bertalanffy2
    (frozen = (; v∞), learning_rate=0.001, half_life=21), # neural
    (frozen = (; v∞), learning_rate=0.001, half_life=21), # neural2
]
iterations = [6000, 6000, 6000, 6000]
comparison = compare(times, volumes, models; calibration_options, iterations)

#-

plot(comparison)

#-

savefig(joinpath(dir, "patientB_validation.png"))
