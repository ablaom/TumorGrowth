using Test

test_names = [
    "tools",
    "functor",
    "patient_data",
    "odes",
    "models",
    "compare", # this must run before "optimisers" and "calibration"
    "optimisers",
    "calibration",
    "integration",
]

names = isempty(ARGS) ? test_names : ARGS

for name in names
    quote
        @testset $name begin
            include($name*".jl")
        end
    end |> eval
end
