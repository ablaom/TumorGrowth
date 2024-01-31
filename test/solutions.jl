using Test
using TumorGrowth
import Lux
import StableRNGs.StableRNG

tolerance = eps()*10^8 # 2.2e-8

times = 1.0:0.1:2.0
v0, v∞, ω, λ = 24.2, 48.2, 0.113, 1.49
p = (; v0, v∞, ω, λ)
abstol = 1e-8
reltol = 1e-8

@testset "`bertalanffy_numerical` and `bertalanffy` agree" begin
    deviations =
        bertalanffy_numerical(times, p; abstol, reltol) - bertalanffy(times, p)
    @test all(abs.(deviations) .< tolerance)
end

@testset "`bertalanffy` and `berta2` ageee when γ = 0" begin
    deviations =
        bertalanffy(times, p) - berta(times, merge(p, (; γ=0.0)); abstol, reltol)
    @test all(abs.(deviations) .< tolerance)
end

@testset "`Neural` objects" begin
    rng = StableRNG(127)
    network = Lux.Dense(2, 2, identity)
    model = neural(rng, network)
    θ = TumorGrowth.initial_parameters(model)
    times = [0.0, 0.01, 0.1, 1.0, 10.0]
    v0, v∞ = 0.00023, 0.00015
    X0 = [v0/v∞, 1.0]

    # solve the underlying linear ode directly:
    A = θ.weight
    @test A*X0 ≈ Lux.apply(network, X0, θ, TumorGrowth.state(model)) |> first
    Xs = map(times) do t
        exp(t*A)*X0
    end
    volumes = first.(Xs) .* v∞

    # compare with applying `Neural` object:
    volumes2 = model(times, (; v0, v∞, θ))

    @test isapprox(volumes, volumes2, rtol=1e-6)

    # The choice of `v∞` matters:
    volumes3 = model(times, (; v0, v∞=100*v∞, θ))
    deviations = volumes2 - volumes3
    @test abs(deviations[3]) > 1e-4
end

true
