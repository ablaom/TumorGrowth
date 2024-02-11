using Test
using TumorGrowth
using IterationControl
import StableRNGs.StableRNG
using Plots

@testset "OptimisationProblem" begin
    f(x) = (x.a[1] - π)^2
    x = (; a=[0.0], b=[42.0])  # second component does not effect `f`
    g(x) = x.a[1] < 10
    err(x) = abs(x.a[1] - π)
    errors = map([0.1, 0.01, 0.001]) do learning_rate
        problem =
            TumorGrowth.OptimisationProblem(f, x; g, learning_rate)
        solve!(problem, 1000)
        solution(problem) |> err
    end
    @test sort(errors) == errors
    @test errors[1] < 10*eps()

    # freeze first component:
    problem = TumorGrowth.OptimisationProblem(f, x; g, frozen=(; a=[0.0],))
    error = solution(problem) |> err
    solve!(problem, 10)
    @test error == solution(problem) |> err

    # freeze second (dummy) component:
    problem = TumorGrowth.OptimisationProblem(f, x; g, frozen=(; b=[42.0]))
    error = solution(problem) |> err
    solve!(problem, 10)
    @test error > solution(problem) |> err

    # increasing scales slow down convergence:
    problem = TumorGrowth.OptimisationProblem(f, x; g)
    solve!(problem, 5)
    loss = IterationControl.loss(problem)
    s(x) = (a = 100*x.a, b=x.b)
    problem = TumorGrowth.OptimisationProblem(f, x; g, scale=s)
    solve!(problem, 5)
    @test loss > IterationControl.loss(problem)

    # going out of bounds:
    gg(x) = x.a[1] < 3
    problem = TumorGrowth.OptimisationProblem(f, x; g=gg)
    @test_logs(
        (:warn, ),
        (:info, ),
        (:info, ),
        solve!(
            problem,
            Step(1),
            InvalidValue(),
            NumberLimit(1000),
        ),
    )
    @test IterationControl.loss(problem) == Inf
    # x still in bounds (not updated last time):
    @test gg(problem.x)
    # x close to boundary:
    @test 3 - solution(problem).a[1] < 0.01

    # out of bounds on construction:
    x = (a=[12.0], b=[42.0])
    @test_throws(
        TumorGrowth.ERR_OUT_OF_BOUNDS,
        TumorGrowth.OptimisationProblem(f, x; g=gg),
    )
end

@testset "CurveOptimisationProblem" begin
    # general quadratic:
    F(x::Real, p) = p.a[1] + p.b[1]*x + p.c[1]*x^2
    F(p) = x -> F(x, p)
    F(xs, p) = F(p).(xs)

    rng = StableRNG(123)
    r = rand(rng, 3)
    p_true = (a=[r[1]], b=[r[2]], c=[r[3]])
    xs = rand(rng, 3)
    ys = F(xs, p_true)
    p0 = (a=[0.0], b=[0.0], c=[0.0])
    threshold = 100*eps()
    problem =
        TumorGrowth.CurveOptimisationProblem(xs, ys, F, p0, learning_rate=0.1)
    outcomes = solve!(
        problem,
        Step(1),
        TimeLimit(0.2/60),
        Threshold(threshold),
    )
    # test that threshold caused the stop:
    @test outcomes[3][2].done

    # smoke test:
    plot(problem)

    p = solution(problem)
    deviations = F(xs, p) - F(xs, p_true)
    @test sum(x->x^2, deviations) < threshold
    @test abs.(p.a[1] - p_true.a[1]) < 0.001
    @test abs.(p.b[1] - p_true.b[1]) < 0.001
    @test abs.(p.c[1] - p_true.c[1]) < 0.001

    # test reproducibility
    problem =
        TumorGrowth.CurveOptimisationProblem(xs, ys, F, p0, learning_rate=0.1)
    solve!(
        problem,
        Step(1),
        TimeLimit(0.2/60),
        Threshold(threshold),
    )
    @test solution(problem) == p
end

true
