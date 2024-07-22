using Test
using TumorGrowth
import StableRNGs.StableRNG
using Lux

# generate some data:
data = patient_data();
record = data[16]
times = record.T_weeks
volumes = record.Lesion_normvol  # volumes normalized by max over dataset

rng = StableRNG(123)
network = Lux.Chain(Dense(1, 2, Lux.tanh; init_weight=Lux.zeros64), Dense(2, 1))
network2 = Lux.Chain(Dense(2, 2, Lux.tanh; init_weight=Lux.zeros64), Dense(2, 2))
n1 = neural(rng, network)
n2 = neural2(rng, network2)

models = [
    logistic,
    gompertz,
    bertalanffy,
    bertalanffy_numerical,
    bertalanffy2,
    n1,
    n2,
]

holdouts = 2
n_iterations = fill(2, length(models))
comparison = compare(times, volumes, models; holdouts, n_iterations)

true
