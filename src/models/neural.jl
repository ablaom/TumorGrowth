mutable struct Neural{O,T,I}
    ode::O
    transform::T
    inverse::I
end

"""
    neural([rng,] network; transform=log, inverse=exp)

Initialize the Lux.jl neural network, `network`, and return a callable object, `model`,
for solving the associated one-dimensional neural ODE for volume growth, as detailed under
"The ODE" below.

!!! important

    Here `network` must accept *one* input and deliver *one* output. For purposes of
    calibration, it may be helpful to use zero-initialization for the first layer. See the
    example below.

The returned object, `model`, is called like this:

    volumes = model(times, p)

where `p` should have properties `v0`, `v∞`, `θ`, where `v0` is the initial volume (so
that `volumes[1] = v0`), `v∞` is a volume scale parameter, and `θ` is a
`network`-compatible Lux.jl parameter. It seems that calibration works best if `v∞` is
frozen.

The form of `θ` is the same as `TumorGrowth.initial_parameters(model)`, which is also the
default initial value used when solving an associated [`CalibrationProblem`](@ref).

```julia
using Lux, Random

# define neural network with 1 input and 1 output:
network = Lux.Chain(Dense(1, 3, Lux.tanh; init_weight=Lux.zeros64), Dense(3, 1))

rng = Xoshiro(123)
model = neural(rng, network)
θ = TumorGrowth.initial_parameters(model)
times = [0.1, 6.0, 16.0, 24.0, 32.0, 39.0]
v0, v∞ = 0.00023, 0.00015
p = (; v0, v∞, θ)

julia> volumes = model(times, p) # (constant because of zero-initialization)
6-element Vector{Float64}:
 0.00023
 0.00023
 0.00023
 0.00023# # Neural2
```

# The ODE

...

See also [`neural2`](@ref), [`TumorGrowth.neural_ode`](@ref).

"""
function neural(args...; transform=log, inverse=exp)
    transform != log && inverse == exp && @warn WARN_TRANSFORM
    inverse(transform(0.1234)) ≈ 0.1234 || @warn WARN_TRANSFORM
    return Neural(neural_ode(args...), transform, inverse)
+end
initial_parameters(model::Neural) = initial_parameters(model.ode)
state(model::Neural) = state(model.ode)

function Base.show(io::IO, ::MIME"text/plain", model::Neural)
    n = Lux.parameterlength(model.ode.θ0)
    print(
        io,
        "Neural model, (times, p) -> volumes, where length(p) = $(n + 2)\n",
        "  transform: $(model.transform)"
    )
end
function Base.show(io::IO, model::Neural)
    n = Lux.parameterlength(model.ode.θ0)
    print(io, "neural ($(n + 2) params)")
end

function (model::Neural)(
    times,
    v0,
    v∞,
    θ;
    saveat = times,
    sensealg = Sens.InterpolatingAdjoint(; autojacvec = Sens.ZygoteVJP()),
    kwargs..., # other `DifferentialEquations.solve` kwargs, eg, `reltol`, `abstol`
    )

    @unpack ode, transform, inverse  = model
    tspan = (times[1], times[end])
    y0 = transform(v0/v∞)
    X0=[y0,]
    problem = DE.ODEProblem(ode, X0, tspan, θ)
    solution = DE.solve(
        problem,
        DE.Tsit5();
        saveat,
        sensealg,
    )
    # return to original scale:
    return v∞*inverse.(first.(solution.u))
end
(model::Neural)(times, p; kwargs...) = model(times, p.v0, p.v∞, p.θ; kwargs...)

function guess_parameters(times, volumes, model::Neural)
        v0 = first(volumes)
        v∞ = sum(volumes)/length(volumes)
        θ = initial_parameters(model)
        return (; v0, v∞, θ)
end

function scale_function(times, volumes, model::Neural)
    volume_scale = sum(volumes)/length(volumes)
    return p ->  (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, θ=p.θ)
end

constraint_function(::Neural) = constraint_function(classical_bertalanffy)

options(::Neural) = (; learning_rate=0.001, frozen=(; v∞=nothing), penalty=0.3)

n_iterations(::Neural) = 2500
