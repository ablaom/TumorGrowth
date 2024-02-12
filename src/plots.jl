const ERR_PLOTS_UNLOADED = ErrorException(
    "To enable plotting you must run `import Plots`. "
)

# these functions extended in /ext/PlotsExt.jl:
plot(args...; kwargs...) = throw(ERR_PLOTS_UNLOADED)
function gui end
