const ERR_MODEL_NOT_VECTOR = ArgumentError(
    "The argument `models` in a call like `ModelComparison(times, volumes, models)` "*
        "must be a vector. "
)
const ERR_MISMATCH = DimensionMismatch(
    "Either `calibration_options` `n_iterations` has a length different from the "*
    "number of models (which must be a vector other iterable). "
)

options(model) =
    (; learning_rate=0.0001, penalty=0.8)
options(model::TumorGrowth.Neural) =
    (; learning_rate=0.01, frozen=(; v∞=nothing))

n_iterations(model) = 10000
n_iterations(::typeof(berta)) = 20000
n_iterations(::TumorGrowth.Neural) = 1500

mae(ŷ, y) = abs.(ŷ .- y) |> mean

struct ModelComparison{T<:Real, MTuple<:Tuple, OTuple<:Tuple, N, M, PTuple<:Tuple}
    times::Vector{T}
    volumes::Vector{T}
    models::MTuple
    n_holdout::Int
    options::OTuple
    n_iterations::NTuple{N,Int64}
    metric::M
    errors::Vector{T}
    parameters::PTuple
    function ModelComparison(
        times::Vector{T},
        volumes::Vector{T},
        models;
        n_holdout=3,
        calibration_options = options.(models),
        n_iterations = n_iterations.(models),
        metric=mae,
        plot=false,
        ) where T<:Real

        length(models) == length(calibration_options) == length(n_iterations) ||
            throw(ERR_MISMATCH)
        _models = Tuple(models)
        _options = Tuple(calibration_options)
        errors, parameters =
            TumorGrowth.errors(
                times,
                volumes,
                _models,
                n_holdout,
                _options,
                n_iterations;
                plot,
            ) # defined below
        MTuple = typeof(_models)
        OTuple = typeof(_options)
        new{T, MTuple, OTuple, length(n_iterations), typeof(metric), typeof(parameters)}(
            times,
            volumes,
            _models,
            n_holdout,
            _options,
            Tuple(n_iterations),
            metric,
            errors,
            parameters,
        )
    end
end

"""
    compare(times, volumes, models; n_holdout=3, metric=mae, advanced_options...)

By calibrating `models` using the specified patient `times` and lesion `volumes`, compare
those models using a hold-out set consisting of the last `n_holdout` data points.

```julia
times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
volumes = [0.00023, 8.4e-5, 6.1e-5, 4.3e-5, 4.3e-5, 4.3e-5]

julia> comparison = compare(times, volumes, [gompertz, logistic])
ModelComparison with 3 holdouts:
  metric: mae
  gompertz:     2.198e-6
  logistic:     6.55e-6

julia> errors(comparison)
2-element Vector{Float64}:
 2.197843662660861e-6
 6.549858321487298e-6

julia> parameters(comparison)[1]
(v0 = 0.00022643603114569068, v∞ = 3.8453274218216947e-5, ω = 0.11537512108224635)
```

# Visualing comparisons

```julia
using Plots
plot(comparison, title="A comparison of two models")
```

# Keyword options

- `n_holdout=3`: number of time-volume pairs excluded from the end of the calibration data

- `metric=mae`: metric applied to holdout set; the reported error on a model predicting
  volumes `v̂` is `metric(v̂, v)` where `v` is the last `n_holdout` values of `volumes`. For
  example, any regression measure from StatisticalMeasures.jl can be used here. The
  built-in fallback is mean absolute error.

- `n_iterations=TumorGrowth.n_iterations.(models)`: a vector of iteration counts for the
  calibration of `models`

- `options=TumorGrowth.options.(models)`: a vector of named tuples providing the keyword
  arguments `CalibrationProblem`s - one for each model. See [`CalibrationProblem`](@ref)
  for details.


See also [`errors`](@ref), [`parameters`](@ref).

"""
compare(args...; kwargs...) = ModelComparison(args...; kwargs...)

"""
    errors(comparison)

Extract the the vector of errors from a `ModelComparison` object, as returned by
calls to [`compare`](@ref).

"""
errors(comparison::ModelComparison) = comparison.errors

"""
    errors(comparison)

Extract the the vector of errors from a `ModelComparison` object, as returned by
calls to [`compare`](@ref).

"""
parameters(comparison::ModelComparison) = comparison.parameters


function errors(etimes, evolumes, models, n_holdout, options, n_iterations; plot=false)

    times = etimes[1:end-n_holdout]
    volumes = evolumes[1:end-n_holdout]

    i = 0
    error_param_pairs = map(models) do model
        i += 1
        n_iter = n_iterations[i]
        problem = CalibrationProblem(times, volumes, model; options[i]...)
        controls = Any[Step(1), InvalidValue(), NumberLimit(n_iter)]
        plot && push!(controls, IterationControl.skip(
            Callback(pr-> (Plots.plot(pr); gui())),
            predicate=div(n_iter, 50),
        ))
        outcomes = solve!(problem, controls...)
        p = solution(problem)
        v̂ = model(etimes, p)
        error = mae(v̂[end-n_holdout+1:end], evolumes[end-n_holdout+1:end])
        (error, p)
    end

    error_tuple, params = zip(error_param_pairs...)
    return collect(error_tuple), params
end

function Base.show(io::IO, comparison::ModelComparison)
    println(io, "ModelComparison with $(comparison.n_holdout) holdouts:")
    print(io, "  metric: $(comparison.metric)")
    for (i, model) in enumerate(comparison.models)
        error = round(comparison.errors[i], sigdigits=4)
        print(io, "\n  $model: \t$error")
    end
    return nothing
end

