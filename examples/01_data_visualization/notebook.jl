using Pkg

dir = @__DIR__
Pkg.activate(dir)
Pkg.instantiate()

import TumorGrowth.patient_data
import DataFrames
using Plots

# # DATA INGESTION

# From the data file, extract a vector of patient records of the form `(id=..., times=...,
# volumes=...)`, one for each patient:
df = patient_data() |> DataFrames.DataFrame
gdf = collect(DataFrames.groupby(df, :Pt_hashID));
patient_records = map(gdf) do sub_df
    (
    id = sub_df[1,:Pt_hashID],
    times = sub_df.T_weeks,
    volumes = sub_df.Lesion_normvol,
    )
end;

# Get the records which have a least 6 measurements:
patient_records6 = filter(patient_records) do s
    length(s.times) >= 6
end;

# Plot some of these records:
plt = plot(xlab="time", ylab="volume (rescaled by maximum)")
selected_indices = [1, 5, 6, 10, 14, 16]
for (i, record) in enumerate(patient_records6[selected_indices])
    times = record.times
    ts = range(times[1], length=40, stop=times[end]) |> collect;
    max = maximum(record.volumes)
    id = record.id[1:4]
    plot!(record.times, (record.volumes) ./ max, label="$id")
    gui()
end
savefig(joinpath(dir, "selected_patient_data.png"))
gui()
