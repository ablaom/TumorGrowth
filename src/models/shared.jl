function DOC_PARAMS(k, ode)
    params = map(sym -> "`$sym`", [:v0, :v∞, :ω, :λ, :γ][1:k])
    params_str = join(params, ", ")
    "Here `p` will have properties $params_str, where `v0` is the "*
    "volume at time `times[1]`"
end

const WARN_TRANSFORM = "`inverse` does not appear to be a true inverse of "*
    "`transform`. Perhaps you specified `transform` but forgot to specify "*
    "`inverse`? "

const DOC_BERTALANFFY_ODE =
    """
    # Underlying ODE

    In the generalized Bertalanffy model, the volume ``v > 0`` evolves according to the
    differential equation

    `` dv/dt = ω B_λ(v_∞/v) v,``

    where ``B_λ`` is the Box-Cox transformation, defined by ``B_λ(x) = (x^λ - 1)/λ``,
    unless ``λ = 0``, in which case, ``B_λ(x) = \\log(x)``. Here:

    - ``v_∞``=`v∞` is the steady state solution, stable and unique, assuming ``ω >
       0``; this is sometimes referred to as the *carrying capacity*

    - ``1/ω`` has the
       units of time

    - ``λ`` is dimensionless

    """

const DOC_SEE_ALSO = "For a list of all models see [`TumorGrowth`](@ref). "

const ERR_UNORDERED_TIMES = ArgumentError("Times must specified in increasing order. ")
