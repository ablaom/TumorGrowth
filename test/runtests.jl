using Test

test_names = [
    "tools",
    "functor",
    "patient_data",
    "odes",
    "solutions",
    "optimisers",
    "calibration",
    "model_comparison",
]

names = isempty(ARGS) ? test_names : ARGS

for name in names
    quote
        @testset $name begin
            include($name*".jl")
        end
    end |> eval
end
