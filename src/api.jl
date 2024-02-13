# A *model* maps times, an initial condition `v0`, and ODE parameters `p` to volumes, as
# in the call `volumes = model(times, p)`.

# Publicly (and below) the parameter `p` a named tuple, although for future proofing,
# assume only that is some property-accessible object. (Internally, e.g., in calibration,
# it is wrapped as a `ComponentArray` because adjoint sensitivity computations applied in
# SciMLSensitivity.jl need parameters to be arrays.)

"""
    guess_parameters(times, volumes, model)

Apply heuristics to guess parameters `p` for a model.

# New model implementations

Fallback returns `nothing` which will prompt user's to explicitly specify initial
parameter values in calibration problems.

"""
guess_parameters(times, volumes, model) = nothing

"""
    scale_function(times, volumes, model)

Return an appropriate function `p -> f(p)` so that `p = f(q)` has a value of the same
order of magnitude expected for `model` parameters, whenever `q` has the same form as `p`
but with all values equal to one.

# New model implementations

Fallback returns the identity.

"""
scale_function(times, volumes, model) = identity

"""
    constraint_function(model)

Return an appropriate `Bool`-valued function `p -> g(p)` which is `false` whenever
parameter `p` leaves the natural domain of `model`.

# New model implementations

Fallback returns `true` always.

"""
constraint_function(model) = _ -> true


"""
    options(model)

Default calibration options for `model` in model comparisons.

# New model implementations

Fallback returns `(learning_rate=0.0001, penalty=0.8)`

"""
options(model) = (learning_rate=0.0001, penalty=0.8)

"""
    n_iterations(model)

Default number of iterations to run calibration of `model` in model comparisons.

# New model implementations

Fallback returns 10000


"""
n_iterations(model) = 10000
