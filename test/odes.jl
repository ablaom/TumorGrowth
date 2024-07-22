using TumorGrowth
import Lux
import StableRNGs.StableRNG

@testset "boxcox" begin
    @test TumorGrowth.boxcox(2, 3) ≈ (3^2 - 1)/2
    @test TumorGrowth.boxcox(0, 3) ≈ log(3)
end

@testset "bertalanffy_ode" begin
    v∞, ω, λ = 2, 7, 0
    v = [10,]
    t = 342.9324
    p = [v∞, ω, λ]
    @test first(TumorGrowth.bertalanffy_ode(v, p, t)) ≈ -7log(5)*10
end

@testset "bertalanffy2_ode!" begin
    p = [1/10, 0, 1/2]
    q = [5*ℯ, 5]
    dq = Vector{Float64}(undef, 2)
    TumorGrowth.bertalanffy2_ode!(dq, q, p, 42)
    @test dq[2] ≈ 1/4
    @test dq[1] ≈ -ℯ/2
end

@testset "neural_ode" begin
    rng = StableRNG(123)
    network = Lux.Chain(Lux.Dense(2, 3, Lux.tanh), Lux.Dense(3, 2))
    ode = TumorGrowth.neural_ode(rng, network)
    θ = TumorGrowth.initial_parameters(ode)
    state = TumorGrowth.state(ode)

    X = rand(rng, 2)
    dX_dt = ode(X, θ, 42.9)
    @test dX_dt ≈ Lux.apply(network, X, θ, state) |> first
end

true
