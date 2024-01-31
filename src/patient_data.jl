const DOC_REF_LALEH =
    "Laleh et al. [(2022)](https://doi.org/10.1371/journal.pcbi.1009822) "*
    "\"Classical mathematical models for prediction of response to "*
    "chemotherapy and immunotherapy\", *PLOS Computational Biology*"

"""
    flat_patient_data()

Return, in row table form, the lesion measurement data collected in $DOC_REF_LALEH.

Each row represents a single measurement of a single lesion on some day.

See also [`patient_data`](@ref), in which each row represents all measurements of a single
lesion.

"""
function flat_patient_data()
    filename = joinpath(@__DIR__, "..", "data", "flat_patient_data.csv")
    return CSV.File(filename) |> Tables.rowtable
end

# script commented out below used to generate "patient_data.csv"

# using TumorGrowth
# import DataFrames
# using CSV
# df = flat_patient_data() |> DataFrames.DataFrame;
# features = names(df)
# gdf = collect(DataFrames.groupby(df, :Pt_hashID));
# records = map(gdf) do sub_df
#     (;
#      Pt_hashID = sub_df[1,:Pt_hashID],
#      Study_Arm = sub_df[1,:Study_Arm],
#      Study_id = sub_df[1,:Study_id],
#      Arm_id = sub_df[1,:Arm_id],
#      T_weeks = sub_df.T_weeks,
#      T_days = sub_df.T_days,
#      Lesion_diam = sub_df.Lesion_diam,
#      Lesion_vol = sub_df.Lesion_vol,
#      Lesion_normvol = sub_df.Lesion_normvol,
#      response = sub_df[1,:response],
#      readings = sub_df[1,:readings],
#      )
# end;
# CSV.write(joinpath(@__DIR__, "..", "data", "patient_data.csv"), records)

vector(str) = eval(Meta.parse(str))
const VEC_FEATURES = [:T_weeks, :T_days, :Lesion_diam, :Lesion_vol, :Lesion_normvol]

"""
    patient_data()

Return, in row table form, the lesion measurement data collected in $DOC_REF_LALEH.

Each row represents all measurements for a single lesion for a unique patient.

```julia
record = first(patient_data())

julia> record.Pt_hashID # patient identifier
"0218075314855e6ceacca856fcd4c737-S1"

julia> record.T_weeks # measure times, in weeks
7-element Vector{Float64}:
  0.1
  6.0
 12.0
 17.0
 23.0
 29.0
 35.0

julia> record.Lesion_normvol # all volumes measured, normalised by dataset max
7-element Vector{Float64}:
 0.000185364052636979
 0.00011229838600811
 8.4371439525252e-5
 8.4371439525252e-5
 1.05464299406565e-5
 2.89394037571615e-5
 8.4371439525252e-5
```

See also [`flat_patient_data`](@ref).

"""
function patient_data()
    filename = joinpath(@__DIR__, "..", "data", "patient_data.csv")
    table = CSV.File(filename) |> Tables.columntable
    for feature in VEC_FEATURES
        f = QuoteNode(feature)
        quote
            $feature = vector.(getproperty($table, $f))
        end |> eval
    end
    parsed = quote
        (; $(VEC_FEATURES...))
    end |> eval
    return merge(table, parsed) |> Tables.rowtable
end
