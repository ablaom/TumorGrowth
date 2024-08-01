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

scale_default(times, volumes, ::typeof(gompertz)) =
    scale_default(times, volumes, classical_bertalanffy)

lower_default(::typeof(gompertz)) = lower_default(classical_bertalanffy)
penalty_default(::typeof(gompertz)) = 0.8
