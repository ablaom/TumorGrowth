# # MODEL API

# A *model* maps times, an initial condition `v0`, and ODE parameters `p` to volumes, as
# in the call `volumes = model(times, p)`.

# Publicly (and below) the parameter `p` is a named tuple, although for future proofing,
# assume only that is some property-accessible object. (Internally, e.g., in calibration,
# it is wrapped as a `ComponentArray` because adjoint sensitivity computations applied in
# SciMLSensitivity.jl need parameters to be arrays.)

const DOC_OPTIMISER = "Here `optimiser` is an optimiser from Optimisers.jl "*
    "(or implements the same API) or is one of: `LevenbergMarquardt()`, "*
    "or `Dogleg()`. "

"""
    guess_parameters(times, volumes, model)

Apply heuristics to guess parameters `p` for a model.

# New model implementations

Fallback returns `nothing` which will prompt user's to explicitly specify initial
parameter values in calibration problems.

"""
guess_parameters(times, volumes, model) = nothing

"""
    scale_default(times, volumes, model)

Return an appropriate function `p -> f(p)` so that `p = f(q)` has a value of the same
order of magnitude expected for `model` parameters, whenever `q` has the same form as `p`
but with all values equal to one.

# New model implementations

Fallback returns the identity.

"""
scale_default(times, volumes, model) = identity


"""
    lower_default(model)

Return a named tuple with the lower bound constraints on `model` parameters.

For example, a return value of `(v0 = 0.1,)` indicates that `p.v0 > 0.1` is a hard
constraint for `p`, in calls of the form `model(times, p)`, but all other components of
`p` are unconstrained.

# New model implementations

Fallback returns `NamedTuple()`.

"""
lower_default(model) = NamedTuple()

"""
    upper_default(model)

Return a named tuple with the upper bound constraints on `model` parameters.

For example, a return value of `(v0 = 1.0,)` indicates that `p.v0 < 1.0` is a hard
constraint for `p`, in calls of the form `model(times, p)`, but all other components of
`p` are unconstrained.

# New model implementations

Fallback returns `NamedTuple()`.

"""
upper_default(model) = NamedTuple()

"""
    options(model, optimiser)

Calibration options for a given `model` and `optimiser`, for use as default options in
model comparisons. $DOC_OPTIMISER.

# New model implementations

Fallback returns:

- `(; Δ=10.0)` if `optimiser isa `LevenbergMarquardt`
- `(; Δ=1.0)` if `optimiser isa `Dogleg`
- `(learning_rate=0.0001, penalty=0.8)` otherwise

"""
options(model, optimiser) = (learning_rate=0.0001, penalty=0.8)
options(model, ::LSO.LevenbergMarquardt) = (; Δ=10.0)
options(model, ::LSO.Dogleg) = (; Δ=1.0)

"""
    n_iterations_default(model, optimiser)

Number of iterations, when calibrating `model` and using `optimiser`, to be adopted by
default in model comparisons. $DOC_OPTIMISER.

# New model implementations

Fallback returns `10000`, unless `optimiser isa Union{LevenbergMarquardt,Dogle}`, in which
case `0` is returned (automatic).

"""
n_iterations_default(model, optimiser) = 10000
n_iterations_default(model, ::GaussNewtonOptimiser) = 0
