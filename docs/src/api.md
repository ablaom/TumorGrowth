# Adding new models

Using a custom model is an advanced option outlined below. This section also serves to aid
developers who want to permanently add new models to the package.

Wherever a model, such as [`bertalanffy`](@ref) or [`gompertz`](@ref), appears in
TumorGrowth.jl workflows, any function or other callable `mymodel` can be used, provided
it has the right signature. The required signature is `mymodel(times, p)`, where `times`
is a vector of times and `p` parameters of the model in the form of a named tuple, such as
`p = (; v0=0.03, ω= 1.5)`. The return value of `mymodel` will be the corresponding volumes
predicted by the model.

If the implementation of `mymodel` requires numerically solving an ordinary differential
equation, follow the example given for the `bertalanffy2` model, which appears
[here](https://github.com/ablaom/TumorGrowth.jl/blob/dev/src/models/bertalanffy2.jl). (In
the TumorGrowth.jl repository, the model ODEs themselves are defined by functions ending
in `_ode` or `_ode!` in a [separate
file](https://github.com/ablaom/TumorGrowth.jl/blob/dev/src/odes.jl).)

Additionally, one may want to overload some of functions listed below for the new model,
especially if convergence during calibration is an issue.  For example, if the new model
is a function is called `mymodel`, and there are two parameters `a` and `b`, then you
overload `guess_parameters` like this:

```julia
function TumorGrowth.guess_parameters(times, volumes, ::typeof(mymodel))
    < code to guess parameters `a` and `b` >
	return (; a=a, b=b)
end 
```

## Optional methods for new model implementations

```@docs
TumorGrowth.guess_parameters
TumorGrowth.scale_function
TumorGrowth.constraint_function
TumorGrowth.options
TumorGrowth.n_iterations
```
