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

warmup(control) = IterationControl.Warmup(control, 500)



# # PATIENT B - RELAPSE FOLLOWING INITIAL IMPROVEMENT

using Lux, Random

network = Chain(
    Dense(2, 5, Lux.tanh, init_weight=Lux.zeros64),
    Dense(5, 2),
)

function normalized_absolute_difference(yhat, y)
    scale = abs(y)
    abs(yhat - y)/scale
end
mape(yhat, y) = sum(broadcast(normalized_absolute_difference, yhat, y))/length(y)

d = Dict()

for (i, record) in enumerate(records6)

    results = Any[]
    for half_life in [Inf, 48, 24, 12]

        etimes = record.times
        evolumes = record.volumes
        times = etimes[1:end-1]
        volumes = evolumes[1:end-1]

        # Note well the zero-initialization of weights in first layer:

        rng = Xoshiro(123)
        model = neural2(rng, network)

        # Note the reduced learning rate.

        v∞ = mean(volumes)

        problem = CalibrationProblem(
            times,
            volumes,
            model;
            scale=identity,
            frozen = (; v∞),
            learning_rate=0.01,
            half_life,
        )

        global losses = Float64[]
        solutions = [solution(problem),]
        solve!(
            problem,
            Step(1),
            InvalidValue(),
            NumberLimit(1500),
            NumberSinceBest(1) |> warmup,
            Callback(pr->push!(solutions, solution(problem))),
            Callback(pr->(plot(pr); gui())) |> sometimes,
            WithLossDo(L->push!(losses, L)),
        )
        losses = losses[100:end]
        plot(losses, title=string(length(losses) + 100))
        gui(); sleep(0.2)

        v̂ = model(etimes, solutions[end-1])[end]
        v = evolumes[end]
        error = mape(v̂, v)
        id = record.id
        push!(results, (; half_life, error, v̂, v, id))

    end
    d[i] = [results...]
end

using Serialization

serialize(joinpath(dir, "effect_of_half_life.jls"), d)
d[1]

errors = Dict()
for half_life in [Inf, 48.0, 24.0, 12.0]
    errors[half_life] = Float64[]
    for i in eachindex(records6)
        results = d[i]
        for row in results
            if row.half_life == half_life
                push!(errors[half_life], row.error)
            end
        end
    end
end

# julia> sum(errors[12.0] .< errors[Inf])/length(records6)
# 0.5491419656786272

# julia> sum(errors[24.0] .< errors[Inf])/length(records6)
# 0.5475819032761311

# julia> sum(errors[48.0] .< errors[Inf])/length(records6)
# 0.5538221528861155

# julia> sum(errors[Inf] .< errors[Inf])/length(records6)
# 0.0
