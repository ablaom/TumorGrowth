"""
    gompertz(times, p)

Return volumes for specified `times`, based on anaytic solutions to the classical Gompertz
model for lesion growth. $(DOC_PARAMS(3, bertalanffy_ode)).

This is the `λ=0` case of the [`bertalanffy`](@ref) model.

$DOC_SEE_ALSO

"""
gompertz(times, p) = bertalanffy(times, TumorGrowth.merge(p, (; λ=0.0)))

guess_parameters(times, volumes, ::typeof(gompertz)) =
    guess_parameters(times, volumes, classical_bertalanffy)

scale_function(times, volumes, model::typeof(gompertz)) =
    scale_function(times, volumes, classical_bertalanffy)

lower(model::typeof(gompertz)) = lower(classical_bertalanffy)
upper(model::typeof(gompertz)) = upper(classical_bertalanffy)
