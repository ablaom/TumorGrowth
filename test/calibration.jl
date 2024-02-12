using Test
using TumorGrowth
import StableRNGs.StableRNG
using IterationControl
import Lux
using Plots

times = [0, 10, 30]
volumes = [3, 6, 15]

@testset "guess_parameters" begin
    # classical:
    p = TumorGrowth.guess_parameters(times, volumes, gompertz)
    @test length(p) == 3
    @test p.v0 ≈ 3
    @test p.v∞ == 15
    @test p.ω ≈ (log(15) - log(3))/(30)

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

    # fallback:
    @test isnothing(TumorGrowth.guess_parameters(times, volumes, "junk"))
end

@testset "scale_function" begin
    # classical:
    s1 = TumorGrowth.scale_function(times, volumes, gompertz)
    scales = s1((v0=1.0, v∞=1.0, ω=1.0))
    @test scales.v0 == 15.0
    @test scales.v∞ == 15.0
    @test scales.ω ≈ abs.(TumorGrowth.guess_parameters(times, volumes, gompertz).ω)

    # one-dimensional TumorGrowth.
    s1 = TumorGrowth.scale_function(times, volumes, bertalanffy)
    scales = s1((v0=1.0, v∞=1.0, ω=1.0, λ=1.0))
    @test scales.v0 == 15.0
    @test scales.v∞ == 15.0
    @test scales.ω ≈ abs.(TumorGrowth.guess_parameters(times, volumes, bertalanffy).ω)
    @test scales.λ == 1.0

    # bertalanffy2
    s1 = TumorGrowth.scale_function(times, volumes, bertalanffy2)
    p = guess_parameters(times, volumes, bertalanffy2)
    vol_scale = p.v∞
    scales = s1((v0=1.0, v∞=1.0, ω=1.0, λ=1.0, γ=1.0))
    @test scales.v0 ≈ vol_scale
    @test scales.v∞ ≈ vol_scale
    @test scales.ω ≈ abs.(TumorGrowth.guess_parameters(times, volumes, bertalanffy2).ω)
    @test scales.λ == 1.0
    @test scales.γ == 1.0

    # fallback:
    @test TumorGrowth.scale_function(times, volumes, "junk") == identity
end

@testset "constraint function" begin
    for model in [gompertz, bertalanffy, bertalanffy2]
        h = TumorGrowth.constraint_function(model)
        p = TumorGrowth.guess_parameters(times, volumes, model)
        @test h(p)
        q = merge(p, (; v∞ = -p.v∞))
        @test !h(q)
        q = merge(p, (; v∞ = -p.v∞, v0 = -p.v0))
        @test !h(q)
        q = merge(p, (; v∞ = -p.v∞))
        @test !h(q)
    end

    h = TumorGrowth.constraint_function("junk")
    @test h("a;lsjfd")
end

function normalized_absolute_difference(yhat, y)
    scale = abs(y)
    abs(yhat - y)/scale
end
mape(yhat, y) = sum(broadcast(normalized_absolute_difference, yhat, y))/length(y)

@testset "CalibrationProblem" begin

    # test calibration using bertalanffy2:
    tol = 0.3
    rng = StableRNG(123)
    p_true = (v0 = 0.013, v∞ = 0.000725, ω = 0.077, λ = 0.2, γ = 1.05)
    times = range(0.1, stop=47.0, length=5) .* (1 .+ .05*rand(rng, 5))
    volumes_true = bertalanffy2(times, p_true)
    problem = CalibrationProblem(
        times,
        volumes_true,
        bertalanffy2;
        frozen = (; λ=p_true[4]),
        learning_rate=0.001,
        penalty=0.01,
        reltol=1e-6,
    )
    l(p) = mape(collect(p), collect(p_true))
    # To diagnose issues, uncomment next 3 commented lines below and do `using Plots`.
    outcomes = solve!(
        problem,
        Step(1),
        InvalidValue(),
        # TimeLimit(5/60),
        NumberLimit(450),
        Callback(problem -> l(solution(problem)) < tol; stop_if_true=true),
        # Callback(problem -> print("\r", pretty(solution(problem)))),
        # Callback(problem->(plot(problem); gui())),
    )
    # check that stop triggered by callback:
    @test outcomes[4][2].done
    @test l(solution(problem)) < tol
    bertalanffy2_loss = loss(problem)
    # smoke test for plots:
    plot(problem)

    # try a neural2 network on the same training data (generated by `bertalanffy2` model) (need
    # 1/10th the learning rate):
    network = Lux.Chain(
        Lux.Dense(2, 5, Lux.tanh, init_weight=Lux.zeros64),
        Lux.Dense(5, 2),
    )
    rng = StableRNG(123)
    model = neural2(rng, network)
    problem = CalibrationProblem(
        times,
        volumes_true,
        model;
        frozen = (; v∞=nothing),
        learning_rate=0.0001,
        half_life=48.0,
    )

    # scatter(times, volumes_true)
    outcomes = solve!(
        problem,
        Step(1),
        InvalidValue(),
        # TimeLimit(5/60),
        NumberLimit(800),
        Threshold(bertalanffy2_loss),
        # Callback(problem -> print("\r", pretty(solution(problem)))),
        # IterationControl.skip(
        #    Callback(problem->(plot(problem); gui())),
        #    predicate=20,),
    )
    # check that stop triggered by threshold:
    @test outcomes[4][2].done
    @test loss(problem) < bertalanffy2_loss

    # check that `v∞` was indeed frozen:
    @test solution(problem).v∞ == TumorGrowth.guess_parameters(times, volumes_true, model).v∞
end

true
