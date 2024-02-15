"""
    logistic(times, v0, v∞, ω)

Return volumes for specified `times`, based on anaytic solutions to the classical logistic
(Verhulst) model for lesion growth. $(DOC_PARAMS(3, bertalanffy_ode)).

This is the `λ=-1` case of the [`bertalanffy`](@ref) model.

$DOC_SEE_ALSO

"""
logistic(times, p) = bertalanffy(times, TumorGrowth.merge(p, (; λ=-1.0)))
const verhulst = logistic

guess_parameters(times, volumes, ::typeof(logistic)) =
    guess_parameters(times, volumes, classical_bertalanffy)

scale_function(times, volumes, model::typeof(logistic)) =
    scale_function(times, volumes, classical_bertalanffy)

constraint_function(model::typeof(logistic)) = constraint_function(classical_bertalanffy)
