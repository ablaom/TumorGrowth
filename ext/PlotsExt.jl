module PlotsExt

using TumorGrowth
using Plots
using UnPack

# Plots.jl linestyle options are [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]

linestyle(::Any) = :auto
linestyle(::typeof(gompertz)) = :dashdot
linestyle(::typeof(classical_bertalanffy)) = :dash
linestyle(::typeof(bertalanffy)) = :solid
linestyle(::typeof(bertalanffy2)) = :dashdotdot
linestyle(::typeof(logistic)) = :dot
linecolor(::Any) = :auto
linecolor(::typeof(gompertz)) = :black
linecolor(::typeof(classical_bertalanffy)) = :black
linecolor(::typeof(bertalanffy)) = :black
linecolor(::typeof(bertalanffy2)) = :black
linecolor(::typeof(logistic)) = :black

@recipe function f(c::TumorGrowth.CurveOptimisationProblem)
    @unpack xs, ys, F, optimisation_problem = c
    xplot = range(xs[1], xs[end], length=200)
    yplot = F(xplot, optimisation_problem.x)
    xplot, yplot
end

@recipe function f(problem::TumorGrowth.CalibrationProblem)
    @unpack volumes, times, model, curve_optimisation_problem = problem
    p = solution(problem)
    L = round(loss(problem), sigdigits=4)
    xlab --> "times"
    ylab --> "volumes"
    title --> "$(pretty(p))\nloss: $L"
    @series begin
        label := "$model"
        seriestype := :path
        xplot = range(times[1], times[end], length=200)
        yplot = model(xplot, p)
        xplot, yplot
    end
    if model == bertalanffy2
        @series begin
            label := "$model (aspirational)"
            seriestype := :path
            linestyle := :dot
            xplot = range(times[1], times[end], length=200)
            yplot = last.(model(xplot, p; aspirational=true))
            xplot, yplot
        end
    end
    @series begin
        label := :none
        seriestype := :scatter
        markershape --> :utriangle
        markercolor --> :black
        times, volumes
    end
end

@recipe function f(comparison::TumorGrowth.ModelComparison)
    etimes = comparison.times
    evolumes = comparison.volumes
    models = comparison.models
    holdouts = comparison.holdouts
    parameters = TumorGrowth.parameters(comparison)
    errors = TumorGrowth.errors(comparison)

    continuous_times = range(etimes[1], stop=etimes[end], length=200)

    xlab --> "time"
    ylab --> "volume"
    @series begin
        seriestype := :scatter
        markershape --> :diamond
        color := :black
        label := :none
        etimes[1:end-holdouts], evolumes[1:end-holdouts]
    end

    @series begin
        seriestype := :scatter
        markershape --> :diamond
        color := :white
        label := "holdout"
        etimes[end-holdouts+1:end], evolumes[end-holdouts+1:end]
    end
    for (i, model) in enumerate(models)
        @series begin
            label := string(model)
            linecolor := linecolor(model)
            linestyle := linestyle(model)
            p = parameters[i]
            continuous_volumes = model(continuous_times, p)
            continuous_times, continuous_volumes
        end
    end
end

end # module
