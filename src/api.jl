# # MODEL API

# A *model* maps times, an initial condition `v0`, and ODE parameters `p` to volumes, as
# in the call `volumes = model(times, p)`.

# Publicly (and below) the parameter `p` is a named tuple, although for future proofing,
# assume only that is some property-accessible object. (Internally, e.g., in calibration,
# it is wrapped as a `ComponentArray` because adjoint sensitivity computations applied in
# SciMLSensitivity.jl need parameters to be arrays.)

const DOC_OPTIMISER = "Here `optimiser` is an optimiser from Optimisers.jl, "*
    "or implements the same API, or is one of: `LevenbergMarquardt()`, "*
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
    lower_default(model)

Return a named tuple with the lower bound constraints on parameters for  `model`.

For example, a return value of `(v0 = 0.1,)` indicates that `p.v0 > 0.1` is a hard
constraint for `p`, in calls of the form `model(times, p)`, but all other components of
`p` are unconstrained.

# New model implementations

Fallback returns `NamedTuple()`.

"""
lower_default(model) = NamedTuple()

"""
    upper_default(model)

Return a named tuple with the upper bound constraints on the parameters for `model`.

For example, a return value of `(v0 = 1.0,)` indicates that `p.v0 < 1.0` is a hard
constraint for `p`, in calls of the form `model(times, p)`, but all other components of
`p` are unconstrained.

# New model implementations

Fallback returns empty named tuple.

"""
upper_default(model) = (;)

"""
    frozen_default(model)

Return a named tuple indicating parameter values to be frozen by default when calibrating
`model`. A value of `nothing` for a parameter indicates freezing at initial value.

# New model implementations

Fallback returns an empty named tuple.

"""
frozen_default(model) = (;)

"""
    optimiser_default(model)

Return the default choice of optimiser for `model`.

# New model implementations

Must return an optimiser from Optimisers.jl, or an optimiser with the same API, or one of
the optimisers from LeastSquaresOptim.jl, such as `LevenbergMarquardt()` or `Dogleg()`.

The fallback returns `Optimisers.Adam(0.0001)`.

"""
optimiser_default(model) = Optimisers.Adam(0.0001)

"""
    scale_default(times, volumes, model)

Return an appropriate default for a function `p -> f(p)` so that `p = f(q)` has a value of
the same order of magnitude expected for parameters of `model`, whenever `q` has the same
form as `p` but with all values equal to one.

Ignored by the optimisers `LevenbergMarquardt()` and `Dogleg()`.

# New model implementations

Fallback returns the identity.

"""
scale_default(times, volumes, model) = identity

"""
    penalty_default(model)

Return the default loss `penalty` to be used when calibrating `model`. The larger the
positive value, the more calibration discourages large differences in `v0` and `v∞`
on a log scale. Helps discourage `v0` and `v∞` drifting out of bounds in models whose
ODE have a singularity at the origin.

Ignored by the optimisers `LevenbergMarquardt()` and `Dogleg()`.

# New model implementations

Must return a value in the range ``[0, ∞)``, and will typically be less than `1.0`. Only
implement if `model` has strictly positive parameters named `v0` and `v∞`.

Fallback returns `0`.

"""
penalty_default(model) = 0

"""
    radius_default(model, optimiser)

Return the default value of `radius` when calibrating `model` using `optimiser`. This is
the initial trust region radius, which is named `Δ` in LeastSquaresOptim.jl documentation
and code.

This parameter is ignored unless `optimiser`, as passed to the `CalibrationProblem`, is
`LevenbergMarquardt()` or `Dogleg()`.

# New model implementations

The fallback returns:

- `10.0` if `optimiser isa `LevenbergMarquardt`
- `1.0` if `optimiser isa `Dogleg`
- `0` otherwise

"""
radius_default(model, optimiser) = 0
radius_default(model, ::LSO.LevenbergMarquardt) = 10.0
radius_default(model, ::LSO.Dogleg) = 1.0

"""
    iterations_default(model, optimiser)

Number of iterations, when calibrating `model` and using `optimiser`, to be adopted by
default in model comparisons. $DOC_OPTIMISER.

# New model implementations

Fallback returns `10000`, unless `optimiser isa Union{LevenbergMarquardt,Dogleg}`, in
which case `0` is returned (stopping controlled by LeastSquaresOptim.jl).

"""
iterations_default(model, optimiser) = 10000
iterations_default(model, ::GaussNewtonOptimiser) = 0
