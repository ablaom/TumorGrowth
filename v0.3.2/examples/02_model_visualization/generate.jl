# execute this julia file to generate the notebooks from notebook.jl

# the figures will need to be manually inserted into the notebook.md file

env = @__DIR__
joinpath(env, "..", "generate.jl") |> include
generate(env, execute=true)


