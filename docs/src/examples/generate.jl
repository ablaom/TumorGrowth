function generate(dir; execute=true, pluto=false)
    quote
        using Pkg
        Pkg.activate(temp=true)
        Pkg.add("Literate")
        using Literate

        OUTDIR = $dir
        outdir = splitpath(OUTDIR)[end]
        INFILE = joinpath(OUTDIR, "notebook.jl")

        @info "Generating notebooks for $outdir. "

        # generate pluto notebook:
        if $pluto
            TEMPDIR = tempdir()
            Literate.notebook(INFILE, TEMPDIR, flavor=Literate.PlutoFlavor())
            mv("$TEMPDIR/notebook.jl", "$OUTDIR/notebook.pluto.jl", force=true)
        else
            @warn "Not generating a Pluto notebook for $outdir."
        end

        Literate.markdown(
            INFILE,
            OUTDIR,
            execute=true,
            # overrides the default ```@example notebook ... ```, which will be ambiguous:
            config=Dict("codefence" => Pair("````@julia", "````" )),
            # config=Dict("codefence" => Pair("````@example $outdir", "````" )),
        )
        # remove if not executing markdown:
        @warn "Any figures in the notebook.md file will need to be manually inserted. "

        Literate.notebook(INFILE, OUTDIR, execute=false)
        mv("$OUTDIR/notebook.ipynb", "$OUTDIR/notebook.unexecuted.ipynb", force=true)
        Literate.notebook(INFILE, OUTDIR, execute=$execute)
        $execute || @warn "Not generating a pre-executed Jupyter notebook for $outdir. "*
            "YOU NEED TO EXECUTE \"notebook.ipynb\" MANUALLY!"

    end |> eval
end
