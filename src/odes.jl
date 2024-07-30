# The odes are here specified in a form suitable for DifferentialEquations.jl

function boxcox(λ, x)
    if x == 0
        λ == 0 && return NaN
        return -1/λ
    end
    x < 0 && return NaN
    λ == 0 && return log(x)
    (x^λ - 1)/λ
end

"""
    bertalanffy_ode(v, p, t)

Based on the generalized Bertalanffy model, return the rate in change in volume at time
`t`, for a current volume of `v`. For details, see [`bertalanffy`](@ref).

Note here that `v`, and the return value, are *vectors* with a single element, rather than
scalars.

"""
function bertalanffy_ode(V, p, t)
    v∞, ω, λ = p
    v = first(V)
    return [ω*boxcox(λ, v∞/v)*v, ]
end

"""
    bertalanffy2_ode!(dX, X, p, t)

A two-dimensional extension of the ODE describing the generalized Bertalanffy model for
lesion growth.  Here `X = [v, u]`, where `v` is volume at time `t` and `u` is the
"carrying capacity" at time `t`, a latent variable. The time derivatives are written to
`dX`. For the specific form of the ODE, see [`bertalanffy2`](@ref).

"""
function bertalanffy2_ode!(dX, X, p, t)
    ω, λ, γ = p
    dX[2] = γ*ω*X[2]
    dX[1] = boxcox(λ, X[2]/X[1])*ω*X[1]
    return nothing
end

mutable struct NeuralODE{N,P,S}
    network::N
    θ0::P
    state::S
end
initial_parameters(ode::NeuralODE) = ode.θ0
state(ode::NeuralODE) = ode.state

"""
   neural_ode([rng,] network)

Initialize the Lux.jl neural2 network, `network`, and return an associated ODE, `ode`, with
calling syntax `dX_dt = ode(X, p, t)`, where `p` is a `network`-compatible parameter.

The initialized parameter value can be recovered with
`TumorGrowth.initial_parameters(ode)`. Get the network state with
`TumorGrowth.state(ode)`.

```julia
using Lux
using Random

rng = Xoshiro(123)
network = network = Lux.Chain(Lux.Dense(2, 3, Lux.tanh), Lux.Dense(3, 2))
ode = neural_ode(rng, network)
θ = TumorGrowth.initial_parameters(ode)
ode(rand(2), θ, 42.9) # last argument irrelevant as `ode` is autonomous
```

"""
function neural_ode(rng, network)
    θ0, state = Lux.setup(rng, network)
    NeuralODE(network, θ0, state)
end
neural_ode(network) = neural_ode(Random.default_rng(), network)

function Base.show(io::IO, ode::NeuralODE)
    n = Lux.parameterlength(ode.θ0)
    print(
        io,
        "NeuralODE, (X, θ, t) -> dX_dt;  θ a $n-dimensional Lux.jl parameter",
    )
end

function (ode::NeuralODE)(X, θ, t)
    dX_dt, state = Lux.apply(ode.network, X, θ, ode.state)
    ode.state = state
    return dX_dt
end
