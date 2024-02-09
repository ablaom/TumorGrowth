# # HELPERS

bertalanffy_analytic_solution(t, v0, v∞, ω, λ) =
    λ == 0 ?
    (v0/v∞)^exp(-ω*t)*v∞ :
    (1 + ((v0/v∞)^λ - 1)*exp(-ω*t))^(1/λ)*v∞


# # FUNCTIONS RETURNING SOLUTIONS TO THE ODES

# A *solution* maps times, an intial condition `v0` and ODE parameters to volumes.

function DOC_PARAMS(k, ode)
    params = map(sym -> "`$sym`", [:v0, :v∞, :ω, :λ, :γ][1:k])
    params_str = join(params, ", ")
    "Here `p` will have properties $params_str, where `v0` is the "*
    "volume at time `times[1]` and the other parameters are explained in "*
    "the [`TumorGrowth.$ode`](@ref) document string"
end

"""
    bertalanffy(times, p)

Return volumes for specified `times`, based on the analytic solution to the generalized
    Bertalanffy model for lesion growth.  $(DOC_PARAMS(4, :bertalanffy_ode)).

See also [`berta`](@ref).

"""
function bertalanffy(times, p)
    @unpack v0, v∞, ω, λ = p
    t0 = first(times)
    b(t) = bertalanffy_analytic_solution(t - t0, v0, v∞, ω, λ)
    return b.(times)
end

"""
    bertalanffy_numerical(times, p; solve_kwargs...)

*Provided for testing purposes.*

Return volumes for specified `times`, based on numerical solutions to the generalized
Bertalanffy model for lesion growth. $(DOC_PARAMS(4, bertalanffy_ode)); `solve_kwargs` are
optional keyword arguments for the ODE solver, `DifferentialEquations.solve`, from
DifferentialEquations.jl.

Since it is based on analtic solutions, [`bertalanffy`](@ref) is the preferred alternative
to this function.

!!! important

    It is assumed without checking that `times` is ordered: `times == sort(times)`.

See also [`berta`](@ref).

"""
function bertalanffy_numerical(
    times,
    p;
    saveat = times,
    sensealg = Sens.InterpolatingAdjoint(; autojacvec = Sens.ZygoteVJP()),
    kwargs..., # other DE.solve kwargs, eg, `reltol`, `abstol`
    )

    @unpack v0, v∞, ω, λ = p

    # We rescale volumes by `v∞` before sending to solver. It is tempting to perform a
    # time-rescaling, but an issue prevents this:
    # https://discourse.julialang.org/t/time-normalisation-results-in-nothing-gradients-of-ode-solutions/109353
    tspan = (times[1], times[end])
    p_ode = [1.0, ω, λ]
    problem = DE.ODEProblem(bertalanffy_ode, v0/v∞, tspan, p_ode)
    solution = DE.solve(problem, DE.Tsit5(); saveat, sensealg, kwargs...)
    # return to original scale:
    return v∞ .* solution.u
end

"""
    berta(times, p; aspirational=false, solve_kwargs...)

Return volumes for specified `times`, based on numerical solutions to a two-dimensional
extension of generalized Bertalanffy model for lesion growth. Here $(DOC_PARAMS(5,
:berta_ode!)).

The usual generalized Bertalanffy model is recovered when `γ=0`. In that case, using
[`bertalanffy`](@ref), which is based on an analytic solution, may be preferred.

!!! important

    It is assumed without checking that `times` is ordered: `times == sort(times)`.

# Keyword options

- `aspirational=false`: Set to `true` to return the aspirational volumes, in addition to
  the actual volumes.

- `solve_kwargs`: optional keyword arguments for the ODE solver,
  `DifferentialEquations.solve`, from DifferentialEquations.jl.

See also [`bertalanffy`](@ref).

"""
function berta(
    times,
    p;
    aspirational=false,
    saveat = times,
    reltol = 1e-7, # this default determined by experiments with patient with id
                 # "44f2f0cc8accfe91e86f0df74346a9d4-S3"; don't raise it without further
                 # investigation.
    sensealg = Sens.InterpolatingAdjoint(; autojacvec = Sens.ZygoteVJP()),
    kwargs..., # other DE.solve kwargs, eg, `reltol`, `abstol`
    )

    @unpack v0, v∞, ω, λ, γ = p

    # We rescale volumes by `v∞` before sending to solver. It is tempting to perform a
    # time-rescaling, but an issue prevents this:
    # https://discourse.julialang.org/t/time-normalisation-results-in-nothing-gradients-of-ode-solutions/109353
    tspan = (times[1], times[end])
    q0 = [v0/v∞, 1.0]
    p = [ω, λ, γ]
    problem = DE.ODEProblem(berta_ode!, q0, tspan, p)
    solution = DE.solve(problem, DE.Tsit5(); saveat, reltol, sensealg, kwargs...)
    # return to original scale:
    aspirational || return v∞ .* first.(solution.u)
    return v∞ .* solution.u
end

_merge(x, y) = merge(x, y)
function _merge(x::ComponentArray, y)
    p, reconstruct = TumorGrowth.functor(x)
    return merge(p, y) |> reconstruct
end

"""
    gompertz(times, p)

Return volumes for specified `times`, based on anaytic solutions to the classical Gompertz
model for lesion growth. $(DOC_PARAMS(3, bertalanffy_ode)).

This is the `λ=0` case of the [`bertalanffy`](@ref) model.

See also [`bertalanffy`](@ref), [`berta`](@ref).

"""
gompertz(times, p) = bertalanffy(times, _merge(p, (; λ=0.0)))

"""
    logistic(times, v0, v∞, ω)

Return volumes for specified `times`, based on anaytic solutions to the classical logistic
(Verhulst) model for lesion growth. $(DOC_PARAMS(3, bertalanffy_ode)).

This is the `λ=-1` case of the [`bertalanffy`](@ref) model.

See also [`bertalanffy`](@ref), [`berta`](@ref).

"""
logistic(times, p) = bertalanffy(times, _merge(p, (; λ=-1.0)))
const verhulst = logistic

"""
    classical_bertalanffy(times, v0, v∞, ω)

Return volumes for specified `times`, based on anaytic solutions to the classical
Bertalanffy model for lesion growth. $(DOC_PARAMS(3, bertalanffy_ode)).

This is the `λ=1/3` case of the [`bertalanffy`](@ref) model.

See also [`bertalanffy`](@ref), [`berta`](@ref).

"""
classical_bertalanffy(times, p) = bertalanffy(times, _merge(p, (; λ=1/3)))

mutable struct Neural{O}
    ode::O
end

"""
    neural([rng,] network)

Initialize the Lux.jl neural network, `network`, and return a callable object, `model`,
for solving the associated neural ODE for volume growth, as detailed under "The ODE"
below.

!!! important

    Here `network` must accept *two* inputs and deliver *two* outputs. For purposes of
    calibration, it may be helpful to use zero-initialization for the first layer. See the
    example below.

The returned object `model` is called like this:

    volumes = model(times, p)

where `p` should have properties `v0`, `v∞`, `θ`, where `v0` is the initial volume (so
that `first(volumes) = v0`), `v∞` is a volume scale parameter, and `θ` is a
`network`-compatible Lux.jl parameter.

The form of `θ` is the same as `TumorGrowth.initial_parameters(model)`, which is also the
default initial value used when solving an associated [`CalibrationProblem`](@ref).

```julia
using Lux, Random

# define neural network with 2 inputs and 2 outputs:
network = Lux.Chain(Dense(2, 3, Lux.tanh; init_weight=Lux.zeros64), Dense(3, 2))

rng = Xoshiro(123)
model = neural(rng, network)
θ = TumorGrowth.initial_parameters(model)
times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
v0, v∞ = 0.00023, 0.00015
p = (; v0, v∞, θ)

julia> volumes = model(times, p) # (constant because of zero-intialization)
6-element Vector{Float64}:
 0.00023
 0.00023
 0.00023
 0.00023
 0.00023
 0.00023
```

# The ODE

...

See also [`TumorGrowth.neural_ode`](@ref).

"""
neural(args...) = Neural(neural_ode(args...))
initial_parameters(model::Neural) = initial_parameters(model.ode)
state(model::Neural) = state(model.ode)

function Base.show(io::IO, ::MIME"text/plain", model::Neural)
    n = Lux.parameterlength(model.ode.θ0)
    print(
        io,
        "Neural model, (times, v0, v∞, θ) -> volumes, where length(θ) = $n",
    )
end
function Base.show(io::IO, model::Neural)
    n = Lux.parameterlength(model.ode.θ0)
    print(io, "neural ($(n + 2) params)")
end

relu(x::T) where T<:Number = x < 0 ? zero(T) : x

function (model::Neural)(
    times,
    v0,
    v∞,
    θ;
    saveat = times,
    sensealg = Sens.InterpolatingAdjoint(; autojacvec = Sens.ZygoteVJP()),
    kwargs..., # other `DifferentialEquations.solve` kwargs, eg, `reltol`, `abstol`
    )
    ode = model.ode
    tspan = (times[1], times[end])
    X0=[v0/v∞, 1.0]
    problem = DE.ODEProblem(ode, X0, tspan, θ)
    solution = DE.solve(
        problem,
        DE.Tsit5();
        saveat,
        sensealg,
    )
    # return to original scale:
    return v∞*relu.(first.(solution.u))
end
(model::Neural)(times, p; kwargs...) = model(times, p.v0, p.v∞, p.θ; kwargs...)
