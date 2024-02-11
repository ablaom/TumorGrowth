using Test

test_names = [
    "tools",
    "functor",
    "patient_data",
    "odes",
    "solutions",
    "model_comparison", # this must run before "optimisers" and "calibration"
    "optimisers",
    "calibration",
]

names = isempty(ARGS) ? test_names : ARGS

for name in names
    quote
        @testset $name begin
            include($name*".jl")
        end
    end |> eval
end
