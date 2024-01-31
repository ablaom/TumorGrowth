using Pkg

dir = @__DIR__
Pkg.activate(dir)
Pkg.instantiate()

import TumorGrowth.bertalanffy
using Plots

times = range(0, 5, length=200)
one = fill(1, length(times))


# # GENERALIZED BERTALANFFY PLOTS: λ = 1/3 (CLASSICAL CASE)

a, b, c, d, e = [1/8, 1/3, 2/3, 1, 5/4, 3/2]
bert_a = bertalanffy(times, a, 1, 1, 1/3);
bert_b = bertalanffy(times, b, 1, 1, 1/3);
bert_c = bertalanffy(times, c, 1, 1, 1/3);
bert_d = bertalanffy(times, d, 1, 1, 1/3);
bert_e = bertalanffy(times, e, 1, 1, 1/3);
labels = map([a, b, c, d, e]) do v0
    r = round(v0, sigdigits=3)
    " V₀=$r"
end |> permutedims
plot(times, hcat(bert_a, bert_b, bert_c, bert_d, bert_e); labels, title="λ=1/3")
savefig(joinpath(dir, "some_solutions1.png"))
gui()


# # GENERALIZED BERTALANFFY PLOTS: V₀=1/3

a, b, c, d, e = [-1, 0, 1/3, 1/2, 1]
bert_a =  bertalanffy(times, 1/3, 1, 1, a);
bert_b = bertalanffy(times, 1/3, 1, 1, b);
bert_c = bertalanffy(times, 1/3, 1, 1, c);
bert_d = bertalanffy(times, 1/3, 1, 1, d);
bert_e = bertalanffy(times, 1/3, 1, 1, e);
labels = map([a, b, c, d, e]) do λ
    r = round(λ, sigdigits=3)
    "λ=$r"
end |> permutedims
plot(times, hcat(bert_a, bert_b, bert_c, bert_d, bert_e); labels, title=" V₀=1/3")
savefig(joinpath(dir, "some_solutions2.png"))
gui()
