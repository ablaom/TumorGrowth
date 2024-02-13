mutable struct Neural2{O}
    ode::O
end

"""
    neural2([rng,] network)

Initialize the Lux.jl neural network, `network`, and return a callable object, `model`,
for solving the associated two-dimensional neural2 ODE for volume growth, as detailed
under "The ODE" below.

!!! important

    Here `network` must accept *two* inputs and deliver *two* outputs. For purposes of
    calibration, it may be helpful to use zero-initialization for the first layer. See the
    example below.

The returned object `model` is called like this:

    volumes = model(times, p)

where `p` should have properties `v0`, `v∞`, `θ`, where `v0` is the initial volume (so
that `volumes[1] = v0`), `v∞` is a volume scale parameter, and `θ` is a
`network`-compatible Lux.jl parameter.

The form of `θ` is the same as `TumorGrowth.initial_parameters(model)`, which is also the
default initial value used when solving an associated [`CalibrationProblem`](@ref).

```julia
using Lux, Random

# define neural network with 2 inputs and 2 outputs:
network = Lux.Chain(Dense(2, 3, Lux.tanh; init_weight=Lux.zeros64), Dense(3, 2))

rng = Xoshiro(123)
model = neural2(rng, network)
θ = TumorGrowth.initial_parameters(model)
times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
v0, v∞ = 0.00023, 0.00015
p = (; v0, v∞, θ)

julia> volumes = model(times, p) # (constant because of zero-initialization)
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

See also [`neural`](@ref), [`TumorGrowth.neural_ode`](@ref).

"""
neural2(args...) = Neural2(neural_ode(args...))
initial_parameters(model::Neural2) = initial_parameters(model.ode)
state(model::Neural2) = state(model.ode)

function Base.show(io::IO, ::MIME"text/plain", model::Neural2)
    n = Lux.parameterlength(model.ode.θ0)
    print(
        io,
        "Neural2 model, (times, v0, v∞, θ) -> volumes, where length(θ) = $n",
    )
end
function Base.show(io::IO, model::Neural2)
    n = Lux.parameterlength(model.ode.θ0)
    print(io, "neural2 ($(n + 2) params)")
end

function (model::Neural2)(
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
    return v∞*first.(solution.u)
end
(model::Neural2)(times, p; kwargs...) = model(times, p.v0, p.v∞, p.θ; kwargs...)

function guess_parameters(times, volumes, model::Neural2)
        v0 = first(volumes)
        v∞ = sum(volumes)/length(volumes)
        θ = initial_parameters(model)
        return (; v0, v∞, θ)
end

function scale_function(times, volumes, model::Neural2)
    volume_scale = sum(volumes)/length(volumes)
    return p ->  (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, θ=p.θ)
end

constraint_function(::Neural2) = constraint_function(classical_bertalanffy)
