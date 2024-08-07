# # Model Battle

# The analysis below is also available in
# [notebook](https://github.com/ablaom/TumorGrowth.jl/tree/dev/docs/src/examples/04_model_battle/)
# form.

# **Note.** The `@threads` cell in this notebook takes about 4 hours to complete on a
# 2018 MacBook Pro:

# We compare the predictive performance of several tumor growth models on data collected
# in Laleh et al. [(2022)](https://doi.org/10.1371/journal.pcbi.1009822) "Classical
# mathematical models for prediction of response to chemotherapy and immunotherapy", *PLOS
# Computational Biology*". In particular, we determine whether differences observed are
# statistically significant.

# In addition to classical models, we include a 2D generalization of the General
# Bertalanffy model, `bertalanffy2`, and some 1D and 2D neural ODE's. The 2D models still
# model a single lesion feature, namely it's volume, but add a second latent variable
# coupled to the volume, effectively making the model second order. For further details,
# refer to the TumorGrowth.jl [package
# documentation](https://ablaom.github.io/TumorGrowth.jl/dev/).

# ## Conclusions

# We needed to eliminate about 2% of patient records because of failure of the neural
# network models to converge before parameters went out of bounds. A bootstrap comparison
# of the differences in mean absolute errors suggest that the General Bertalanffy model
# performs significantly better than all other models, with of the exception the 1D neural
# ODE. However, in pair-wise comparisons the neural ODE model was *not* significantly
# better than any model. Results are summarised in the table below. Arrows point to
# bootstrap winners in the top row or first column.

# |                 model | gompertz | logistic | classical_bertalanffy | bertalanffy | bertalanffy2 |   n1 | n2 |
# |-----------------------|----------|----------|-----------------------|-------------|--------------|------|----|
# |           exponential |        ↑ |     draw |                     ↑ |           ↑ |         draw | draw |  ← |
# |              gompertz |      n/a |     draw |                  draw |           ↑ |         draw | draw |  ← |
# |              logistic |     draw |      n/a |                  draw |           ↑ |         draw | draw |  ← |
# | classical_bertalanffy |     draw |     draw |                   n/a |           ↑ |         draw | draw |  ← |
# |           bertalanffy |        ← |        ← |                     ← |         n/a |            ← | draw |  ← |
# |          bertalanffy2 |     draw |     draw |                  draw |           ↑ |          n/a | draw |  ← |
# |                    n1 |     draw |     draw |                  draw |        draw |         draw |  n/a |  ← |

# This demonstration has been tested in the Julia package environment specified by the
# Project.toml and Manifest.toml files located
# [here](https://github.com/ablaom/TumorGrowth.jl/tree/dev/docs/src/examples/04_model_battle).

using Pkg #hide
dir = @__DIR__ #hide
Pkg.activate(dir) #hide
Pkg.instantiate() #hide

using Random
using Statistics
using TumorGrowth
using Lux
using Plots
import PrettyPrint.pprint
using PrettyTables
using Bootstrap
using Serialization
using ProgressMeter
using .Threads


# ## Data ingestion

# Collect together all records with at least 6 measurements, from the data

records = filter(patient_data()) do record
    record.readings >= 6
end;

# Here's what a single record looks like:

pprint(records[13])


# ## Neural ODEs

# We define some one and two-dimensional neural ODE models we want to include in our
# comparison. The choice of architecture here is somewhat ad hoc and further
# experimentation might give better results.

network = Chain(
    Dense(1, 3, Lux.tanh, init_weight=Lux.zeros64),
    Dense(3, 1),
)

network2 = Chain(
    Dense(2, 2, Lux.tanh, init_weight=Lux.zeros64),
    Dense(2, 2),
)

n1 = neural(Xoshiro(123), network) # `Xoshiro` is a random number generator
n2 = neural2(Xoshiro(123), network2)


# ## Models to be compared

model_exs =
    [:exponential, :gompertz, :logistic, :classical_bertalanffy, :bertalanffy,
     :bertalanffy2, :n1, :n2]
models = eval.(model_exs)


# ## Computing prediction errors on a holdout set

holdouts = 2
errs = fill(Inf, length(records), length(models))

p = Progress(length(records))

@threads for i in eachindex(records)
    record = records[i]
    times, volumes = record.T_weeks, record.Lesion_normvol
    comparison = compare(times, volumes, models; holdouts)
    errs[i,:] = TumorGrowth.errors(comparison)
    next!(p)
end
finish!(p)

#-

serialize(joinpath(dir, "errors.jls"), errs);

# ## Bootstrap comparisons (neural ODE's excluded)

# Because the neural ODE errors contain more `NaN` values, we start with a comparison that
# excludes them, discarding only those observations where `NaN` occurs in a non-neural
# model.

bad_error_rows = filter(axes(errs, 1)) do i
    es = errs[i,1:end-2]
    any(isnan, es) || any(isinf, es) || max(es...) > 0.1
end
proportion_bad = length(bad_error_rows)/size(errs, 1)
@show proportion_bad

# That's less than 0.5%. Let's remove them:

good_error_rows = setdiff(axes(errs, 1), bad_error_rows);
errs = errs[good_error_rows,:];

# Errors are evidently not normally distributed (and we were not able to transform them
# to approximately normal):

plt = histogram(errs[:, 1], normalize=:pdf, alpha=0.4)
histogram!(errs[:, end-2], normalize=:pdf, alpha=0.4)

#-

savefig(joinpath(dir, "errors_distribution.png"))

# We deem a student t-test inappopriate and instead compute bootstrap confidence intervals
# for pairwise differences in model errors:

confidence_intervals = Array{Any}(undef, length(models) - 2, length(models) - 2)
for i in 1:(length(models) - 2)
    for j in 1:(length(models) - 2)
        b = bootstrap(
            mean,
            errs[:,i] - errs[:,j],
            BasicSampling(10000),
        )
        confidence_intervals[i,j] = only(confint(b, BasicConfInt(0.95)))[2:3]
    end
end
confidence_intervals

# We can interpret the confidence intervals as  follows:

# - if both endpoints -ve, row index wins

# - if both endpoints +ve, column index wins

# - otherwise a draw

winner_pointer(ci) = ci == (0, 0) ? "n/a" :
    isnan(first(ci)) && isnan(last(ci)) ? "inconclusive" :
    first(ci) < 0 && last(ci) < 0 ? "←" :
    first(ci) > 0 && last(ci) > 0 ? "↑" :
    "draw"

tabular(A, model_exs) = NamedTuple{(:model, model_exs[2:end]...)}((
    model_exs[1:end-1],
    (A[1:end-1, j] for j in 2:length(model_exs))...,
))

pretty_table(
    tabular(winner_pointer.(confidence_intervals), model_exs[1:5]),
    show_subheader=false,
)

# ## Bootstrap comparison of errors (neural ODE's included)

bad_error_rows = filter(axes(errs, 1)) do i
    es = errs[i,:]
    any(isnan, es) || any(isinf, es) || max(es...) > 0.1
end
proportion_bad = length(bad_error_rows)/size(errs, 1)
@show proportion_bad

# We remove the additional 2%:

good_error_rows = setdiff(axes(errs, 1), bad_error_rows);
errs = errs[good_error_rows,:];

# And proceed as before, but with all columns of `errs` (all models):

confidence_intervals = Array{Any}(undef, length(models), length(models))
for i in 1:length(models)
    for j in 1:length(models)
        b = bootstrap(
            mean,
            errs[:,i] - errs[:,j],
            BasicSampling(10000),
        )
        confidence_intervals[i, j] = only(confint(b, BasicConfInt(0.95)))[2:3]
    end
end

pretty_table(
    tabular(winner_pointer.(confidence_intervals), model_exs),
    show_subheader=false,
    tf=PrettyTables.tf_markdown, vlines=:all,
)

# The lack of statistical significance notwithstanding, here are the models, listed in
# order of decreasing performance:

zipped = collect(zip(models, vec(mean(errs, dims=1))))
sort!(zipped, by=last)
model, error = collect.(zip(zipped...))
rankings = (; model, error)
pretty_table(
    rankings,
    show_subheader=false,
    tf=PrettyTables.tf_markdown, vlines=:all,
)
