using Test
using TumorGrowth
import StableRNGs.StableRNG
using IterationControl
using Statistics
using ComponentArrays
import Optimisers

# generate some data:
rng = StableRNG(123)
ptrue = (v0 = 0.013, v∞ = 0.000725, ω = 0.077, λ = 0.2)
times = range(0.1, stop=47.0, length=8) .* (1 .+ .05*rand(rng, 8))
volumes = bertalanffy(times, ptrue) #.* (1 .+ .05*rand(rng, 8))

models =  [bertalanffy,             logistic, gompertz]
holdouts = 2
options = [(; optimiser=Dogleg()), (;),       (;)]
n_iters = [nothing,                nothing,   3]
errs, ps = TumorGrowth.errors(
    times,
    volumes,
    models,
    holdouts,
    options,
    n_iters,
    false,
)

# # compute errors by hand

# 1. bertalanffy (Dogleg)
problem=CalibrationProblem(times[1:end-2], volumes[1:end-2], bertalanffy; optimiser=Dogleg())
solve!(problem, 0)
p = solution(problem)
v̂ = bertalanffy(times, p)
err_bertalanffy = mean(abs.(v̂[end-1:end] - volumes[end-1:end]))

# 2. logistic (Adam)
problem = CalibrationProblem(times[1:end-2], volumes[1:end-2], logistic)
n_iter = TumorGrowth.iterations_default(logistic, "adam")
solve!(problem, Step(1), InvalidValue(), NumberLimit(n_iter))
p = solution(problem)
v̂ = logistic(times, p)
err_logistic = mean(abs.(v̂[end-1:end] - volumes[end-1:end]))

# 3. gompertz (Adam, iterations specified)
problem=CalibrationProblem(times[1:end-2], volumes[1:end-2], gompertz)
solve!(problem, 3)
p = solution(problem)
v̂ = gompertz(times, p)
err_gompertz = mean(abs.(v̂[end-1:end] - volumes[end-1:end]))

# # compare with `TumorGrowth.errors` result:
# @test ComponentArray(ps[2]) ≈  ComponentArray(p)
@test errs[1] ≈ err_bertalanffy
@test errs[2] ≈ err_logistic
@test errs[3] ≈ err_gompertz

# integration:
comparison = compare(
    times,
    volumes,
    models;
    holdouts,
    calibration_options=options,
    iterations=n_iters,
)
@test comparison.iterations ==
    (0, TumorGrowth.iterations_default(
        logistic,
        TumorGrowth.optimiser_default(logistic),
        ),
     3)
@test ComponentArray(parameters(comparison)[1]) ≈ ComponentArray(ptrue)
@test errors(comparison)[2] ≈ err_logistic

# smoke tests for plots:
@test_throws(
    TumorGrowth.ERR_PLOTS_UNLOADED,
    compare(times, volumes, models; holdouts, plot=true),
)

using Plots
comparison = compare(times, volumes, models; holdouts, plot=true)
plot(comparison)

true
