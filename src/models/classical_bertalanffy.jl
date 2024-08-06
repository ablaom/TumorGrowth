"""
    classical_bertalanffy(times, v0, v∞, ω)

Return volumes for specified `times`, based on anaytic solutions to the classical
Bertalanffy model for lesion growth. $(DOC_PARAMS(3, bertalanffy_ode)).

This is the `λ=1/3` case of the [`bertalanffy`](@ref) model.

$DOC_SEE_ALSO

"""
classical_bertalanffy(times, p) = bertalanffy(times, TumorGrowth.merge(p, (; λ=1/3)))

function guess_parameters(times, volumes, ::typeof(classical_bertalanffy))
    v0 = first(volumes)
    non_zero_volumes = filter(v -> v > eps(float(eltype(volumes))), volumes)
    isempty(non_zero_volumes) && throw(ERR_VOLUMES_TOO_SMALL)
    v∞ = last(non_zero_volumes)
    if length(non_zero_volumes) == 1
        ω = inv(last(times) - first(times))
    else
        τ1 = min(times...)
        τ2 = max(times...)
        v1 = min(non_zero_volumes...)
        v2 = max(volumes...)
        ω = (log(v2) - log(v1))/(τ2 - τ1)
    end
    return (; v0, v∞, ω)
end

function scale_default(times, volumes, model::typeof(classical_bertalanffy))
    p = guess_parameters(times, volumes, model)
    volume_scale = abs(p.v∞)
    time_scale = 1/abs(p.ω)
    return p -> (v0=volume_scale*p.v0, v∞=volume_scale*p.v∞, ω=p.ω/time_scale)
end

lower_default(model::typeof(classical_bertalanffy)) = (v0=0, v∞=0)
penalty_default(::typeof(classical_bertalanffy)) = 0.8
