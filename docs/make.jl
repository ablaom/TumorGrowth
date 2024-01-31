using Documenter
using TumorGrowth

makedocs(;
         modules=[TumorGrowth,],
         format=Documenter.HTML(),
         pages=[
             "Quick start" => "index.md",
             "Reference" => "reference.md",
             ],
         sitename="TumorGrowth",
         warnonly = [:cross_references, :missing_docs],
)

deploydocs(
    repo = "github.com/ablaom/TumorGrowth.jl",
    devbranch="dev",
    push_preview=false,
)
