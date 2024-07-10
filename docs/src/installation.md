# Installation

TumorGrowth.jl can be installed in any Julia [package
environment](https://docs.julialang.org/en/v1/stdlib/Pkg/). When trying TumorGrowth.jl out
for the first time, we recommend installing it in a fresh environment, as shown below.

```julia
using Pkg
Pkg.activate("my_oncology_project", shared=true)
Pkg.add("TumorGrowth")
Pkg.add("Plots")
```

You should now be able to run the code in the [Quick start](@ref) or Extended
examples sections.

In a new julia session you can re-activate the environment created above with:

```julia
using Pkg
Pkg.activate("my_oncology_project", shared=true)
```


