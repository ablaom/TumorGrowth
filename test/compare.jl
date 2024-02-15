using Test
using TumorGrowth
import StableRNGs.StableRNG
using IterationControl
using Statistics
using ComponentArrays

# generate some data:
rng = StableRNG(123)
p = (v0 = 0.013, v∞ = 0.000725, ω = 0.077, λ = 0.2, γ = 1.05)
times = range(0.1, stop=47.0, length=5) .* (1 .+ .05*rand(rng, 5))
volumes = gompertz(times, p) .* (1 .+ .05*rand(rng, 5))

models = [logistic, bertalanffy]
holdouts = 2
options = TumorGrowth.options.(models)
n_iters = TumorGrowth.n_iterations.(models)
errs, ps =
    TumorGrowth.errors(times, volumes, models, holdouts, options, n_iters, false, false)

# compute the `bertalanffy` error by hand:
problem = CalibrationProblem(times[1:end-2], volumes[1:end-2], bertalanffy; options[2]...)
solve!(problem, Step(1), InvalidValue(), NumberLimit(n_iters[2]))
p = solution(problem)
v̂ = bertalanffy(times, p)
err = mean(abs.(v̂[end-1:end] - volumes[end-1:end]))

# and compare with `TumorGrowth.errors`:
@test ComponentArray(ps[2]) ≈  ComponentArray(p)
@test errs[2] ≈ err

# integration:
comparison = compare(times, volumes, models; holdouts)
@test ComponentArray(parameters(comparison)[2]) ≈ ComponentArray(p)
@test errors(comparison)[2] ≈ err

# smoke tests for plots:
@test_throws(
    TumorGrowth.ERR_PLOTS_UNLOADED,
    compare(times, volumes, models; holdouts, plot=true),
)
using Plots
comparison = compare(times, volumes, models; holdouts, plot=true)
plot(comparison)

true
