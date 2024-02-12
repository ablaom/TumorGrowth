const ERR_P0_UNSPECIFIED = ArgumentError( "Unable to infer initial parameter value. You
        must specify `p0=...`. " )

const ERR_UNRECOGNIZED_KEY = ArgumentError(
        "In `frozen` you have specified an unrecognized parameter name. "
)


# # HEURISTICS TO GUESS INITIAL PARAMETER VALUES AND THEIR SCALES

const BertalanffyModel =
        Union{typeof(bertalanffy),typeof(bertalanffy_numerical)}
const ClassicalModel =
    Union{typeof(gompertz),typeof(logistic),typeof(classical_bertalanffy)}
const NeuralModel = Union{Neural,Neural2}

guess_parameters(times, volumes, model) = nothing
function guess_parameters(times, volumes, ::ClassicalModel)
    v0 = first(volumes)
    non_zero_volumes = filter(v -> v > eps(float(eltype(volumes))), volumes)
    isempty(non_zero_volumes) &&
        error("All provided volumes are too small for meaningful calibration. ")
    v∞ = last(non_zero_volumes)
    τ1 = min(times...)
    τ2 = max(times...)
    v1 = min(non_zero_volumes...)
    v2 = max(volumes...)
    ω = (log(v2) - log(v1))/(τ2 - τ1)
    return (; v0, v∞, ω)
end
guess_parameters(times, volumes, ::BertalanffyModel) =
        merge(guess_parameters(times, volumes, gompertz), (; λ=1/3))
function guess_parameters(times, volumes, ::typeof(bertalanffy2))
    κ = 0.5*sign(TumorGrowth.curvature(times, volumes))
    fallback =  merge(guess_parameters(times, volumes, bertalanffy), (; γ=κ))

    problem = CalibrationProblem(
        times,
        volumes,
        bertalanffy;
        learning_rate=0.0001,
        penalty=1.0,
    )

    try
        outcomes = solve!(problem, Step(1), InvalidValue(), NumberLimit(1000))
        outcomes[2][2].done && return fallback # out of bounds
        return merge(solution(problem), (; γ=κ))
    catch
        return fallback
    end

end
function guess_parameters(times, volumes, model::NeuralModel)
        v0 = first(volumes)
        v∞ = sum(volumes)/length(volumes)
        θ = initial_parameters(model)
        return (; v0, v∞, θ)
end

scale_function(times, volumes, model) = identity
function scale_function(times, volumes, model::ClassicalModel)
    p = guess_parameters(times, volumes, model)
    volume_scale = abs(p.v∞)
    time_scale = 1/abs(p.ω)
    return p -> (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, ω=p.ω/time_scale)
end
function scale_function(times, volumes, model::BertalanffyModel)
    p = guess_parameters(times, volumes, model)
    volume_scale = abs(p.v∞)
    time_scale = 1/abs(p.ω)
    return p -> (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, ω=p.ω/time_scale, λ=p.λ)
end
function scale_function(times, volumes, model::typeof(bertalanffy2))
    p = guess_parameters(times, volumes, model)
    volume_scale = abs(p.v∞)
    time_scale = 1/abs(p.ω)
    p ->
        (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, ω=p.ω/time_scale, λ=p.λ, γ=p.γ)
end
function scale_function(times, volumes, model::NeuralModel)
        volume_scale = sum(volumes)/length(volumes)
        return p ->  (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, θ=p.θ)
end


# # CONSTRAINT FUNCTIONS

constraint_function(model) = _ -> true
constraint_function(model::Union{
    ClassicalModel,
    BertalanffyModel,
    NeuralModel,
    typeof(Bertalanffy2),
}) = p -> p.v0 > 0 && p.v∞ > 0


# # PENALTY HELPER

@inline function factor(p, penalty)
    a, b = p.v0, p.v∞
    return ((a^2 + b^2)/(2a*b))^penalty
end


# # OPTIMIZATION PROBLEM FOR A SINGLE PATIENT RECORD

mutable struct CalibrationProblem{T,M,C}
    times::Vector{T}
    volumes::Vector{T}
    model::M
    curve_optimisation_problem::C
end


function Base.show(io::IO, problem::CalibrationProblem)
    p = pretty(solution(problem))
    print(io, "CalibrationProblem: \n  "*
        "model: $(problem.model)\n  "*
        "current solution: $p"
          )
    return
end


"""
        CalibrationProblem(times, volumes, model; learning_rate=0.0001, options...)

Specify a problem concerned with optimizing the parameters of a tumor growth `model`,
given measured `volumes` and corresponding `times`.

Here `model` can be one of: `bertalanffy`, `bertalanffy_numerical`, `bertalanffy2`, `gompertz`,
`logistic`, `classical_bertalanffy`.

Default optimisation is by Adam gradient descent, using a sum of squares loss. Call
`solve!` on a problem to carry out optimisation, as shown in the example below. See
"Extended Help" for advanced options, including early stopping.

Initial values of the parameters are inferred by default.

# Simple solve

```julia
using TumorGrowth

times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
volumes = [0.00023, 8.4e-5, 6.1e-5, 4.3e-5, 4.3e-5, 4.3e-5]
problem = CalibrationProblem(times, volumes, gompertz; learning_rate=0.01)
solve!(problem, 30)    # apply 30 gradient descent updates
julia> loss(problem)   # sum of squares loss
1.7341026729860452e-9

p = solution(problem)
julia> pretty(p)
"v0=0.0002261  v∞=2.792e-5  ω=0.05731"


extended_times = vcat(times, [42.0, 46.0])
julia> gompertz(extended_times, p)[[7, 8]]
2-element Vector{Float64}:
 3.374100207406809e-5
 3.245628908921241e-5
```

# Extended help

# Solving with iteration controls

Continuing the example above, we may replace the number of iterations, `n`, in
`solve!(problem, n)`, with any control from IterationControl.jl:

```julia
using IterationControl
solve!(
  problem,
  Step(1),            # apply controls every 1 iteration...
  WithLossDo(),       # print loss
  Callback(problem -> print(pretty(solution(problem)))), # print parameters
  InvalidValue(),     # stop for ±Inf/NaN loss, incl. case of out-of-bound parameters
  NumberSinceBest(5), # stop when lowest loss so far was 5 steps prior
  TimeLimit(1/60),    # stop after one minute
  NumberLimit(400),   # stop after 400 steps
)
p = solution(problem)
julia> loss(problem)
7.609310030658547e-10
```

See [IterationControl.jl](https://github.com/JuliaAI/IterationControl) for all options.

# Visualizing results

```julia
using Plots
scatter(times, volumes, xlab="time", ylab="volume", label="train")
plot!(problem, label="prediction")
```

# Models

A `model` can be any callable object returning volumes given times and parameters, as in
`predicted_volumes = model(times, p)`. TumorGrowth.jl supplies the following built-in
models:

| model                           | description                             | parameters, `p`       | analytic? | ODE                                     |
|:--------------------------------|:----------------------------------------|:----------------------|:----------|:----------------------------------------|
| [`bertalanffy`](@ref)           | generalized Bertalanffy (GB)            | `(; v0, v∞, ω, λ)`    | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`bertalanffy_numerical`](@ref) | generalized Bertalanffy (testing only)  | `(; v0, v∞, ω, λ)`    | no        | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`bertalanffy2`](@ref)          | 2D extension of generalized Bertalanffy | `(; v0, v∞, ω, λ, γ)` | no        | [`TumorGrowth.bertalanffy2_ode!`](@ref) |
| [`gompertz`](@ref)              | Gompertz (GB, `λ=0`)                    | `(; v0, v∞, ω)`       | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`logistic`](@ref)              | logistic/Verhulst (GB, `λ=-1`)          | `(; v0, v∞, ω)`       | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`classical_bertalanffy`](@ref) | classical Bertalanffy (GB, `λ=1/3`)     | `(; v0, v∞, ω)`       | yes       | [`TumorGrowth.bertalanffy_ode`](@ref)   |
| [`neural2(rng, network)`](@ref) | neural2 ODE with Lux.jl `network`       | `(; v0, v∞, θ)`       | no        | [`TumorGrowth.neural_ode`](@ref)        |

In every case `v0` is the initial volume (so that `predicted_volumes[1] == v0`). Note this
can be different from `volumes[1]`.

# Keyword options

- `p0`: initial value of the model parameters (always a vector); inferred by default for
  built-in models

- `g=(p-> true)`: constraint function: If `g(p) == false` for some parameter `p`, then a
  warning is given and `solution(problem)` is frozen at the last constrained value of `p`;
  use `solve!(problem, Step(1), InvalidValue(), ...)` to ensure early stopping (which
  works because `IterationControl.loss(problem)` will return `Inf` in that case). If
  unspecified, the constraint function is inferred in the case of built-in models and
  parameters are otherwise unconstrained.

- `frozen`: a named tuple, such as `(; v0=nothing, λ=1/2)`; indicating parameters to be
  frozen at specified values during optimization; a `nothing` value means freeze at
  initial value.

- `learning_rate=0.0001`: learning rate for Adam gradient descent optimiser

- `optimiser=Optimisers.Adam(learning_rate)`: optimiser; must be from Optimisers.jl.

- `scale`: a scaling function with the property that `p = scale(q)` has a value of the
  same order of magnitude for the problem parameters being optimised, whenever `q` has the
  same form as `p` but with all values equal to one. Scaling ensures gradient descent
  learns all components of `p` at a similar rate. If unspecified, scaling is inferred for
  built-in models, and uniform otherwise.


- `half_life=Inf`: set to a real positive number to replace the sum of squares loss with a
  weighted version; weights decay in reverse time with the specified `half_life`

- `penalty=0.0` (range=``[0, ∞)``): the larger the positive value, the more a loss
  function modification discourages large differences in `v0` and `v∞` on a log
  scale. Helps discourage `v0` and `v∞` drifting out of bounds in models whose ODE have a
  singularity at the origin (all built-in models except the neural network models).

- `ode_options...`: optional keyword arguments for the ODE solver,
  `DifferentialEquations.solve`, from DifferentialEquations.jl. Not relevant for models
  using analytic solutions (see table above).

"""
function CalibrationProblem(
    times,
    volumes,
    model;
    p0=guess_parameters(times, volumes, model),
    scale=scale_function(times, volumes, model),
    g=constraint_function(model),
    frozen = NamedTuple(),
    half_life = Inf,
    penalty = 0.0,
    learning_rate=0.0001,
    optimiser=Optimisers.Adam(learning_rate),
    ode_options...,
    )

    times = collect(times)

    # zero volume not allowed:
    T = float(eltype(volumes))
    volumes = map(volumes) do v
        v < eps(T) ? eps(T) : v
    end

    times == sort(times) || error("The supplied times are not in increasing order. ")

    # fill in `nothing` values of `frozen` with values from `p0`:
    actual_frozen = TumorGrowth.recover(frozen, p0)

    # determine actual starting `p`
    isnothing(p0) && throw(ERR_P0_UNSPECIFIED)
    names = keys(p0)
    frozen_names = keys(actual_frozen)
    frozen_names ⊆ names || throw(ERR_UNRECOGNIZED_KEY)
    p = merge(p0, actual_frozen) |> ComponentArray

    function actual_scale(p)
        _p, reconstruct = TumorGrowth.functor(p)
        return reconstruct(scale(_p))
    end

    l2 = WeightedL2Loss(times, half_life)
    if penalty == 0
        loss = l2
    elseif [:v0, :v∞] ⊆ keys(p)
        loss(ŷ, y, p) = l2(ŷ, y)*factor(p, penalty)
    else
        @warn "Ignoring `penalty`. "
        loss =l2
    end

    cop = CurveOptimisationProblem(
        times,
        volumes,
        model,
        p;
        g,
        frozen=actual_frozen,
        scale=actual_scale,
        loss,
        learning_rate,
        optimiser,
        ode_options...,
    )

    return CalibrationProblem(times, volumes, model, cop)
end

"""
    CalibrationProblem(problem; kwargs...)

Construct a new calibration problem out an existing `problem` but supply new keyword
arguments, `kwargs`. Unspecified keyword arguments fall back to defaults, except for `p0`,
which falls back to `solution(problem)`.

"""
CalibrationProblem(problem::CalibrationProblem; p0 = solution(problem), kwargs...) =
    CalibrationProblem(problem.times, problem.volumes, problem.model; p0, kwargs...)

IterationControl.loss(c::CalibrationProblem) =
    IterationControl.loss(c.curve_optimisation_problem)
function IterationControl.train!(c::CalibrationProblem, n)
    problem = c.curve_optimisation_problem
    solution = IterationControl.train!(problem, n)
    c.curve_optimisation_problem = problem
    return solution
end

"""
    solution(problem)

Return to the solution to a [`CalibrationProblem`](@ref). Normally applied after calling
[`solve!(problem)`](@ref).


Also returns the solution to internally defined problems, as constructed with
`TumorGrowth.OptimisationProblem`, `TumorGrowth.CurveOptimisationProblem`.

"""
solution(c::CalibrationProblem) =
    solution(c.curve_optimisation_problem)
