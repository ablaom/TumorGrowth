using Test
using TumorGrowth
using IterationControl
using StableRNGs
using ComponentArrays
import LeastSquaresOptim as LSO
import Optimisers

@testset "OptimisationProblem" begin
    f(x) = (x.a - π)^2
    x = (; a=0.0, b=42.0) |>  ComponentArray # second component does not effect `f`
    lower = (; )
    upper = (; a = 10)
    err(x) = abs(x.a - π)
    errors = map([0.1, 0.01, 0.001]) do learning_rate
        opt = Optimisers.Adam(learning_rate)
        problem =
            TumorGrowth.OptimisationProblem(
                f,
                x,
                lower,
                upper,
                opt,
                (;),      # `frozen`
                identity, # `scale`
        )
        solve!(problem, 1000)
        solution(problem) |> err
    end
    @test sort(errors) == errors
    @test errors[1] < 10*eps()

    opt = Optimisers.Adam(0.1)

    # freeze first component:
    problem = TumorGrowth.OptimisationProblem(f, x, lower, upper, opt, (; a=0.0), identity)
    error = solution(problem) |> err
    solve!(problem, 10)
    @test error == solution(problem) |> err

    # freeze second (dummy) component:
    problem = TumorGrowth.OptimisationProblem(f, x, lower, upper, opt,(; b=42.0), identity)
    error = solution(problem) |> err
    solve!(problem, 10)
    @test error > solution(problem) |> err

    # increasing scaless slow down convergence:
    problem = TumorGrowth.OptimisationProblem(f, x, lower, upper, opt, (;), identity)
    solve!(problem, 5)
    L = IterationControl.loss(problem)
    s(x) = (a = 100*x.a, b=x.b)
    problem = TumorGrowth.OptimisationProblem(f, x, lower, upper, opt, (;), s)
    solve!(problem, 5)
    @test L > IterationControl.loss(problem)

    # going out of bounds:
    upper = (; a=3)
    problem = TumorGrowth.OptimisationProblem(f, x, lower, upper, opt, (;), s)
    @test solve!(problem, 1000) ≈ ComponentArray(a=3, b=42.0)
    # @test_logs(
    #     (:warn, ),
    #     (:info, ),
    #     (:info, ),
    #     solve!(
    #         problem,
    #         Step(1),
    #         InvalidValue(),
    #         NumberLimit(1000),
    #     ),
    # )
    # @test IterationControl.loss(problem) == Inf
    # # x still in bounds (not updated last time):
    # @test gg(problem.x)
    # # x close to boundary:
    # @test 3 - solution(problem).a[1] < 0.01

    # out of bounds on construction:
    x = (a=12.0, b=42.0)
    @test_throws(
        TumorGrowth.ERR_OUT_OF_BOUNDS,
        TumorGrowth.OptimisationProblem(f, x, lower, upper, opt, (;), identity),
    )
end

# general quadratic:
F(x::Real, p) = p.a[1] + p.b[1]*x + p.c[1]*x^2
F(p) = x -> F(x, p)
F(xs, p) = F(p).(xs)

rng = StableRNG(123)
r = rand(rng, 3)
p_true = (a=[r[1]], b=[r[2]], c=[r[3]])
xs = rand(rng, 3)
ys = F(xs, p_true)
p0 = (a=[0.0], b=p_true.b, c=[0.0])
threshold = 100*eps()

@testset "GaussNewtonProblem" begin
    frozen = (; b=p_true.b)
    pfree, reconstruct = TumorGrowth.functor(p0, frozen)
    cfree = ComponentArray(pfree)
    h(cfree) = F(xs, reconstruct(cfree))
    f!(out, cfree) = copy!(out, h(cfree) .- ys)
    function g!(J, cfree)
        for i in eachindex(xs)
            J[i, 1] = 1
            J[i, 2] = xs[i]^2
        end
    end

    least_squares_problem =
        LSO.LeastSquaresProblem(; x = cfree, f!, g!, output_length = length(xs))

    problem = TumorGrowth.GaussNewtonProblem(
        least_squares_problem,
        (;),  # lower
        (;),  # upper,
        10.0, # Δ 
        LSO.LevenbergMarquardt(),
        reconstruct,
    )

    # smoke test
    solve!(problem, 1)
    @test loss(problem) < Inf

    # let LSO control stopping:
    solve!(problem)
    solution(problem)
    p = solution(problem)
    deviations = F(xs, p) - F(xs, p_true)
    @test sum(x->x^2, deviations) < threshold
    @test p.a[1] ≈ p_true.a[1]
    @test p.b[1] ≈ p_true.b[1]
    @test p.c[1] ≈ p_true.c[1]
end

@testset "CurveOptimisationProblem: gradient descent" begin
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

@testset "CurveOptimisationProblem: Gauss-Newton" begin
    problem =
        TumorGrowth.CurveOptimisationProblem(
            xs,
            ys,
            F,
            p0,
            optimiser = LSO.LevenbergMarquardt(),
        )
    solve!(problem, 0)

    p = solution(problem)
    deviations = F(xs, p) - F(xs, p_true)
    @test sum(x->x^2, deviations) < threshold
    @test abs.(p.a[1] - p_true.a[1]) < 0.001
    @test abs.(p.b[1] - p_true.b[1]) < 0.001
    @test abs.(p.c[1] - p_true.c[1]) < 0.001
end

true
