pretty(p) = repr(p)
function pretty(p::Union{
    NamedTuple{(:v0, :v∞, :ω)},
    NamedTuple{(:v0, :v∞, :ω, :λ)},
    NamedTuple{(:v0, :v∞, :ω, :λ, :γ)},
    })
    q = map(r->round(r; sigdigits=3), collect(p))
    output = "v0=$(q[1])  v∞=$(q[2])  ω=$(q[3])"
    k = length(p)
    k > 3 || return output
    output *= "  λ=$(q[4])"
    k > 4 || return output
    output *= "  γ=$(q[5])"
    return output
end

const NeuralParameter = NamedTuple{(:v0, :v∞, :θ)}
function pretty(p::NeuralParameter)
    @unpack v0, v∞, θ = p
    Θ = ComponentArray(θ)
    μ = sum(Θ)/length(Θ)
    v0, v∞, μ = map((v0, v∞, μ)) do r
        round(r, sigdigits=4)
    end
    return "v0=$v0  v∞=$v∞  mean(θ)=$μ"
end
