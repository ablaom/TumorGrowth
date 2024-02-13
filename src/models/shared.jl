function DOC_PARAMS(k, ode)
    params = map(sym -> "`$sym`", [:v0, :v∞, :ω, :λ, :γ][1:k])
    params_str = join(params, ", ")
    "Here `p` will have properties $params_str, where `v0` is the "*
    "volume at time `times[1]` and the other parameters are explained in "*
    "the [`TumorGrowth.$ode`](@ref) document string"
end

const WARN_TRANSFORM = "`inverse` does not appear to be a true inverse of "*
    "`transform`. Perhaps you specified `transform` but forgot to specify "*
    "`inverse`? "
