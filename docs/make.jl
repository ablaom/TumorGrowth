using Documenter
using TumorGrowth

makedocs(;
         modules=[TumorGrowth,],
         format=Documenter.HTML(
             size_threshold_ignore = [
                 "examples/03_calibration/notebook.md",
                 "examples/04_model_battle/notebook.md",
             ]
         ),
         pages=[
             "Overview" => "index.md",
             "Installation" => "installation.md",
             "Quick start" => "quick_start.md",
             "Calibration" => "calibration.md",
             "Model comparison" => "comparison.md",
             "Extended examples" => [
                 "Calibration workflows" => "examples/03_calibration/notebook.md",
                 "Model battle" => "examples/04_model_battle/notebook.md",
             ],
             "Adding new models" => "api.md",
             "Reference" => "reference.md",
             ],
         sitename="TumorGrowth.jl",
         warnonly = [:cross_references, :missing_docs],
)

deploydocs(
    repo = "github.com/ablaom/TumorGrowth.jl",
    devbranch="dev",
    push_preview=false,
)
