using Pkg
dir = @__DIR__
Pkg.activate(dir)
Pkg.instantiate()

using Statistics
import DataFrames
using Plots
using IterationControl
using TumorGrowth

Plots.scalefontsizes() # reset font sizes
Plots.scalefontsizes(0.85)


# # DATA INGESTION

# From the data file, extract a vector of patient records of the form `(id=..., times=...,
# volumes=...)`, one for each patient:
df = patient_data() |> DataFrames.DataFrame;
gdf = collect(DataFrames.groupby(df, :Pt_hashID));
records = map(gdf) do sub_df
    (
    id = sub_df[1,:Pt_hashID],
    times = sub_df.T_weeks,
    volumes = sub_df.Lesion_normvol,
    )
end;

# Get the records which have a least 6 measurements:
records6 = filter(records) do s
    length(s.times) >= 6
end;


# # HELPERS

# Wrapper to only apply control every 100 steps:

sometimes(control) = IterationControl.skip(control, predicate=100)

# Wrapper to only apply control after first 30 steps:

warmup(control) = IterationControl.Warmup(control, 30)


# # PATIENT A - A VOLUME THAT IS MOSTLY DECREASIING

record = records6[2] # records6[3] a problem
gui()

#-

times = record.times
volumes = record.volumes;

# We'll try calibrating the generalized Bertalanffy model, `bertalanffy`, with fixed
# parameter `λ=1/5`:

problem = CalibrationProblem(
    times,
    volumes,
    bertalanffy;
    frozen=(; λ=1/5),
    learning_rate=0.001,
    half_life=21, # place greater weight on recent measurements
)

solve!(
    problem,
    Step(1),             # compute 1 iteration at a time
    InvalidValue(),      # to catch parameters going out of bounds
    NumberLimit(6000),   # stop after 4000 steps
    # TimeLimit(1/60),   # stop after 1 minute
    GL() |> warmup,      # stop using Prechelt's GL criterion
    NumberSinceBest(10)  |> warmup, # stop when it's 10 steps since the best so far
    Callback(prob-> (plot(prob); gui())) |> sometimes,
    # WithLossDo(),
)
gui()

#-

p = solution(problem)
extended_times = vcat(times, [40.0, 47.0])
bertalanffy(extended_times, p)

#-

plot(problem, title="bertalanffy, λ=1/5 fixed")
gui()

#-

savefig(joinpath(dir, "patientA.png"))

#-


# # PATIENT B - RELAPSE FOLLOWING INITIAL IMPROVEMENT

record = records6[10]
gui()

times = record.times
volumes = record.volumes

# We'll first try the earlier simple model:

problem = CalibrationProblem(
    times,
    volumes,
    bertalanffy;
    frozen = (; λ=1/5),
    learning_rate=0.001,
    half_life=21, # place greater weight on recent measurements
)

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
)
plot(problem, label="bertalanffy, λ fixed")
gui()

# Let's try the 2D generalization of the TumorGrowth.model, still fixing `λ`:

problem = CalibrationProblem(
    times,
    volumes,
    berta;
    frozen = (; λ=1/5),
    learning_rate=0.001,
    half_life=21,
)

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
)
plot!(problem, label="berta, λ fixed")
gui()

#-

# And finally, we'll try a neural ODE model, with fixed volume scale `v∞`.

using Lux, Random

# Note well the zero-initialization of weights in first layer:

network = Chain(
    Dense(2, 5, Lux.tanh, init_weight=Lux.zeros64),
    Dense(5, 2),
)
rng = Xoshiro(123)
model = neural(rng, network)

n = Lux.parameterlength(network)

# Note the reduced learning rate.

v∞ = mean(volumes)

problem = CalibrationProblem(
    times,
    volumes,
    model;
    scale=identity,
    frozen = (; v∞),
    learning_rate=0.001,
    half_life=21,
)

solve!(
    problem,
    Step(1),
    InvalidValue(),
    NumberLimit(6000),
)
plot!(
    problem,
    title = "Model comparison for a patient",
    label = "neural",
    legend=:inside,
)
gui()

#-

savefig(joinpath(dir, "patientB.png"))
