const ERR_MODEL_NOT_VECTOR = ArgumentError(
    "The argument `models` in a call like `ModelComparison(times, volumes, models)` "*
        "must be a vector. "
)
const ERR_MISMATCH = DimensionMismatch(
    "Either `calibration_options` `n_iterations` has a length different from the "*
    "number of models (which must be a vector other iterable). "
)

mae(ŷ, y) = abs.(ŷ .- y) |> mean

struct ModelComparison{T<:Real,MTuple<:Tuple,OTuple<:Tuple,N,M,PTuple<:Tuple}
    times::Vector{T}
    volumes::Vector{T}
    models::MTuple
    holdouts::Int
    options::OTuple
    n_iterations::NTuple{N,Int64}
    metric::M
    errors::Vector{T}
    parameters::PTuple
    function ModelComparison(
        times::Vector{T},
        volumes::Vector{T},
        models;
        holdouts=3,
        calibration_options=fill((;), length(models)),
        n_iterations=fill(nothing, length(models)),
        metric=mae,
        plot=false,
        ) where T<:Real

        length(models) == length(calibration_options) == length(n_iterations) ||
            throw(ERR_MISMATCH)
        _models = Tuple(models)
        _options = Tuple(calibration_options)
        errors, parameters, actual_iterations =
            TumorGrowth.errors(
                times,
                volumes,
                _models,
                holdouts,
                _options,
                n_iterations,
                plot,
            ) # defined below
        MTuple = typeof(_models)
        OTuple = typeof(_options)
        new{T, MTuple, OTuple, length(n_iterations), typeof(metric), typeof(parameters)}(
            times,
            volumes,
            _models,
            holdouts,
            _options,
            actual_iterations,
            metric,
            errors,
            parameters,
        )
    end
end

"""
    compare(times, volumes, models; holdouts=3, metric=mae, advanced_options...)

By calibrating `models` using the specified patient `times` and lesion `volumes`, compare
those models using a hold-out set consisting of the last `holdouts` data points.

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

julia> p = parameters(comparison)[1]  # calibrated parameter for `gompertz`
(v0 = 0.00022643603114569068, v∞ = 3.8453274218216947e-5, ω = 0.11537512108224635)

julia> gompertz(times, p)
6-element Vector{Float64}:
 0.00022643603114569068
 9.435316392754094e-5
 5.1039159299783234e-5
 4.303209015899451e-5
 4.021112910411027e-5
 3.922743006690166e-5
```

When a model parameter becomes out of bounds, calibration stops early and the last
in-bounds value is reported.

# Visualising comparisons

```julia
using Plots
plot(comparison, title="A comparison of two models")
```

# Keyword options

- `holdouts=3`: number of time-volume pairs excluded from the end of the calibration data

- `metric=mae`: metric applied to holdout set; the reported error on a model predicting
  volumes `v̂` is `metric(v̂, v)` where `v` is the last `holdouts` values of `volumes`. For
  example, any regression measure from StatisticalMeasures.jl can be used here. The
  built-in fallback is mean absolute error.

- `n_iterations=TumorGrowth.n_iterations.(models)`: a vector of iteration counts for the
  calibration of `models`

- `calibration_options`: a vector of named tuples providing keyword arguments for the
  `CalibrationProblem` for each model. Possible keys are: `p0`, `lower`, `upper`,
  `frozen`, `learning_rate`, `optimiser`, `radius`, `scale`, `half_life`, `penalty`, and
  keys corresponding to any ODE solver options. Keys left unspecified fall back to
  defaults, as these are described in the [`CalibrationProblem`](@ref) document string.

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
    parameters(comparison)

Extract the the vector of parameters from a `ModelComparison` object, as returned by
calls to [`compare`](@ref).

"""
parameters(comparison::ModelComparison) = comparison.parameters

function errors(
    etimes,
    evolumes,
    models,
    holdouts,
    options,
    n_iterations,
    plot,
    )
    times = etimes[1:end-holdouts]
    volumes = evolumes[1:end-holdouts]

    # initialize what will be the actual number of iterations used:
    actual_iterations = Int[]

    i = 0
    error_param_pairs = map(models) do model
        i += 1
        problem = CalibrationProblem(times, volumes, model;  options[i]...)
        optimiser = TumorGrowth.optimiser(problem)
        n_iter = isnothing(n_iterations[i]) ? n_iterations_default(model, optimiser) :
            n_iterations[i]
        push!(actual_iterations, n_iter)
        if n_iter > 0
            step =
                optimiser isa GaussNewtonOptimiser ? Step(n_iter) : Step(1)
            number_limit =
                optimiser isa GaussNewtonOptimiser ? NumberLimit(1) : NumberLimit(n_iter)
            predicate =
                optimiser isa GaussNewtonOptimiser ? 1 : div(n_iter, 50)
            controls = Any[step, InvalidValue(), number_limit]
            plot && push!(controls, IterationControl.skip(
                Callback(pr-> (TumorGrowth.plot(pr); TumorGrowth.gui()));
                predicate,
            ))
            solve!(problem, controls...)
        else
            solve!(problem, 0)
            plot && (TumorGrowth.plot(problem); gui())
        end
        p = solution(problem)
        v̂ = model(etimes, p)
        error = mae(v̂[end-holdouts+1:end], evolumes[end-holdouts+1:end])
        (error, p)
    end

    error_tuple, params = zip(error_param_pairs...)
    return collect(error_tuple), params, Tuple(actual_iterations)
end

function Base.show(io::IO, comparison::ModelComparison)
    println(io, "ModelComparison with $(comparison.holdouts) holdouts:")
    print(io, "  metric: $(comparison.metric)")
    for (i, model) in enumerate(comparison.models)
        error = round(comparison.errors[i], sigdigits=4)
        print(io, "\n  $model: \t$error")
    end
    return nothing
end
