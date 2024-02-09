using StableRNGs
using TumorGrowth
using IterationControl

tol = 0.001
rng= StableRNG(127)
p_true = rand(rng, 5)
times = rand(rng, 5) |> sort
volumes_true = bertalanffy2(times, p_true)
problem = CalibrationProblem(
    times,
    volumes_true;
    frozen = (; λ=p_true[4]),
    model=bertalanffy2,
    learning_rate=0.1,
)
l(p) = maximum(abs.(p_true - p))

@time solve!(
    problem,
    Step(1),
    InvalidValue(),
    TimeLimit(1/120),
    Callback(p-> l(p) < tol; stop_if_true=true),
)

# ForwardDiff 5.3s; 51155 iterations
# Zygote:     14s;38332 iteration
