using Test
using TumorGrowth
import Lux
import StableRNGs.StableRNG

times = [0, 10, 30]
volumes = [3, 6, 15]

@testset "guess_parameters" begin
    # classical:
    p = TumorGrowth.guess_parameters(times, volumes, gompertz)
    @test length(p) == 3
    @test p.v0 ≈ 3
    @test p.v∞ == 15
    @test p.ω ≈ (log(15) - log(3))/(30)
    @test_throws(
        TumorGrowth.ERR_VOLUMES_TOO_SMALL,
        TumorGrowth.guess_parameters(times, [0, 0, 0], gompertz),
    )
    @test TumorGrowth.guess_parameters(times, [0, 1, 0], gompertz).ω ≈ 1/30

    # one-dimensional TumorGrowth.
    p = TumorGrowth.guess_parameters(times, volumes, bertalanffy)
    @test length(p) == 4
    @test p.v0 ≈ 3
    @test p.v∞ == 15
    @test p.ω ≈ (log(15) - log(3))/(30)
    @test p.λ ≈ 1/3

    # two-d TumorGrowt (just a smoke test).
    p = TumorGrowth.guess_parameters(times, volumes, bertalanffy2)
    @test length(p) == 5
    keys(p) == (:v∞, :ω, :λ, :γ)

    # exponential:
    p = TumorGrowth.guess_parameters(times[1:2], exp.(times[1:2]), exponential)
    @test length(p) == 2
    @test p.v0 ≈ exp(times[1])
    @test p.ω ≈ -1.0
    @test_throws(
        TumorGrowth.ERR_VOLUMES_TOO_SMALL,
        TumorGrowth.guess_parameters(times, [0, 0, 0], exponential),
    )
    @test TumorGrowth.guess_parameters(times, [0, 1, 0], exponential).ω ≈ 1/30

    # fallback:
    @test isnothing(TumorGrowth.guess_parameters(times, volumes, "junk"))
end

@testset "scale_default" begin
    # classical:
    s1 = TumorGrowth.scale_default(times, volumes, gompertz)
    scales = s1((v0=1.0, v∞=1.0, ω=1.0))
    @test scales.v0 == 15.0
    @test scales.v∞ == 15.0
    @test scales.ω ≈ abs.(TumorGrowth.guess_parameters(times, volumes, gompertz).ω)

    # one-dimensional TumorGrowth:
    s1 = TumorGrowth.scale_default(times, volumes, bertalanffy)
    scales = s1((v0=1.0, v∞=1.0, ω=1.0, λ=1.0))
    @test scales.v0 == 15.0
    @test scales.v∞ == 15.0
    @test scales.ω ≈ abs.(TumorGrowth.guess_parameters(times, volumes, bertalanffy).ω)
    @test scales.λ == 1.0

    # bertalanffy2:
    s1 = TumorGrowth.scale_default(times, volumes, bertalanffy2)
    p = TumorGrowth.guess_parameters(times, volumes, bertalanffy2)
    vol_scale = p.v∞
    scales = s1((v0=1.0, v∞=1.0, ω=1.0, λ=1.0, γ=1.0))
    @test scales.v0 ≈ vol_scale
    @test scales.v∞ ≈ vol_scale
    @test scales.ω ≈ abs.(TumorGrowth.guess_parameters(times, volumes, bertalanffy2).ω)
    @test scales.λ == 1.0
    @test scales.γ == 1.0

    # exponential:
    s1 = TumorGrowth.scale_default(times[1:2], exp.(2*times[1:2]), exponential)
    scales = s1((v0=1.0, ω=1.0))
    @test scales.v0 == exp(2*times[1])
    @test scales.ω ≈ 2.0/log(2)

    # fallback:
    @test TumorGrowth.scale_default(times, volumes, "junk") == identity
end

@testset "lower and upper defaults" begin
    for model in [gompertz, bertalanffy, bertalanffy2]
        lower = TumorGrowth.lower_default(model)
        upper = TumorGrowth.upper_default(model)
        p = TumorGrowth.guess_parameters(times, volumes, model)
        @test TumorGrowth.satisfies_constraints(p, lower, upper)
        p = merge(p, (; v∞ = -p.v∞))
        @test !TumorGrowth.satisfies_constraints(p, lower, upper)
        p = merge(p, (; v∞ = -p.v∞, v0 = -p.v0))
        @test !TumorGrowth.satisfies_constraints(p, lower, upper)
        p = merge(p, (; v∞ = -p.v∞))
        @test !TumorGrowth.satisfies_constraints(p, lower, upper)
    end
    model = exponential
    lower = TumorGrowth.lower_default(model)
    upper = TumorGrowth.upper_default(model)
    @test TumorGrowth.satisfies_constraints((; v0=1, ω=10), lower, upper)
    @test !TumorGrowth.satisfies_constraints((; v0=-1, ω=100), lower, upper)
    @test isempty(TumorGrowth.lower_default("junk"))
    @test isempty(TumorGrowth.upper_default("junk"))
end

tolerance = eps()*10^8 # 2.2e-8

times = 1.0:0.1:2.0
v0, v∞, ω, λ = 24.2, 48.2, 0.113, 1.49
p = (; v0, v∞, ω, λ)
abstol = 1e-8
reltol = 1e-8

@testset "bertalanffy_analytic_solution NaN logic" begin
    f = TumorGrowth.bertalanffy_analytic_solution
    @test f(42, 0, v∞, ω, λ) == 0
    @test f(42, v0, 0, ω, λ) |> isnan
    @test f(42, -v0, v∞, ω, λ) |> isnan
    @test f(42, v0, -v∞, ω, λ) |> isnan
    @test f(42, v0, v∞, ω, 0) ≈ (v0/v∞)^exp(-ω*42)*v∞
    @test f(42, v0, v∞, -2, λ) |> isnan
end

@testset "`bertalanffy_numerical` and `bertalanffy` agree" begin
    deviations =
        bertalanffy_numerical(times, p; abstol, reltol) - bertalanffy(times, p)
    @test all(abs.(deviations) .< tolerance)
end

@testset "`bertalanffy` and `bertalanffy2` ageee when γ = 0" begin
    deviations =
        bertalanffy(times, p) - bertalanffy2(times, merge(p, (; γ=0.0)); abstol, reltol)
    @test all(abs.(deviations) .< tolerance)
end

@testset "`exponential`" begin
    @test exponential(times, (; v0, ω)) ≈ v0*exp.(-ω*(times .- times[1]))
end

@testset "`Neural2` objects" begin
    rng = StableRNG(127)
    network = Lux.Dense(2, 2, identity)
    model = neural2(rng, network; transform=identity, inverse=identity)
    # hack to kill off the bias (Lux.jl v1.0 no-longer zeros the bias):
    old = model.ode.θ0
    model.ode.θ0 = (; weight=old.weight, bias=zeros(Float32, size(old.bias)...))
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

    # compare with applying `Neural2` object:
    volumes2 = model(times, (; v0, v∞, θ))

    @test isapprox(volumes, volumes2, rtol=1e-6)

    # The choice of `v∞` matters:
    volumes3 = model(times, (; v0, v∞=100*v∞, θ))
    deviations = volumes2 - volumes3
    @test abs(deviations[3]) > 1e-4
end

true
