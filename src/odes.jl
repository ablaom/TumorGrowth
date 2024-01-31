# The odes are here specified in a form suitable for DifferentialEquations.jl

function boxcox(λ, x)
    λ == 0 && return log(x)
    (x^λ - 1)/λ
end

"""
    bertalanffy_ode(v, p, t)

Based on the general TumorGrowth.model, return the rate in change in volume at time `t`,
for a current volume of `v`. This first order, autonomous, one-dimensionaly ODE model is
given by

`` dv/dt = ω((v∞/v)^λ - 1)/λ)v; v>0``

where `[v∞, ω, λ] == p` are paramters::

-  ``v∞`` is the steady state solution, stable and unique, assuming ``ω > 0``
- ``1/ω`` has the units of time
- ``λ`` is dimensionless

When ``λ = -1``, one recovers the logistic (Verhulst) model, while ``λ = 1/3`` gives the
classical TumorGrowth.model.  In the case ``λ = 0``, the implementation replaces
``((v∞/v)^λ - 1)/λ`` with its limiting value ``\\log(v∞/v)`` to recover the Gompertz
model.

"""
function bertalanffy_ode(v, p, t)
    v∞, ω, λ = p
    return ω*boxcox(λ, v∞/v)*v
end

"""
    berta_ode!(dX, X, p, t)

A two-dimensional extension of the generalized TumorGrowth.model for lesion growth (see
[`bertalanffy_ode`](@ref)). Here `X = [v, u]`, where `v` is volume at time `t` and `u` is
the "aspirational volume" at time `t`, a latent variable. The time derivatives are written
to `dX`. Specifically, `dX` will have these components:

`` dv/dt = ω((u/v)^λ - 1)/λ)v``
`` du/dt = γωu; ``


where `[ω, λ, γ] == p` are fixed parameters:

- ``1/ω`` has units of time
- ``λ`` is dimensionless
- ``γ`` is dimensionless

When ``γ = 0`` the model collapses to the (one-dimensional) TumorGrowth.model. In that
special case, ``λ = -1``, gives the logistic (Verhulst) model, while ``λ = 1/3`` gives the
classical TumorGrowth.model.  In the case ``λ = 0``, the implementation replaces
``((u/v)^λ - 1)/λ`` with its limiting value ``\\log(u/v)`` to recover the Gompertz
model when also ``γ = 0``.

Since `u` is a latent variable, its initial value is an additional model parameter.

"""
function berta_ode!(dX, X, p, t)
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

Initialize the Lux.jl neural network, `network`, and return an associated ODE, `ode`, with
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
