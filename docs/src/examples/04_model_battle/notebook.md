```@meta
EditURL = "notebook.jl"
```

# Model Battle

The analysis below is also available in
[notebook](https://github.com/ablaom/TumorGrowth.jl/tree/dev/docs/src/examples/04_model_battle/)
form.

**Note.** The `@threads` cell in this notebook takes about 4 hours to complete on a
2018 MacBook Pro:

We compare the predictive performance of several tumor growth models on data collected
in Laleh et al. [(2022)](https://doi.org/10.1371/journal.pcbi.1009822) "Classical
mathematical models for prediction of response to chemotherapy and immunotherapy", *PLOS
Computational Biology*". In particular, we determine whether differences observed are
statistically significant.

In addition to classical models, we include a 2D generalization of the General
Bertalanffy model, `bertalanffy2`, and some 1D and 2D neural ODE's. The 2D models still
model a single lesion feature, namely it's volume, but add a second latent variable
coupled to the volume, effectively making the model second order. For further details,
refer to the TumorGrowth.jl [package
documentation](https://ablaom.github.io/TumorGrowth.jl/dev/).

## Conclusions

We needed to eliminate about 2% of patient records because of failure of the neural
network models to converge before parameters went out of bounds. A bootstrap comparison
of the differences in mean absolute errors suggest that the General Bertalanffy model
performs significantly better than all other models, with of the exception the 1D neural
ODE. However, in pair-wise comparisons the neural ODE model was *not* significantly
better than any model. Results are summarised in the table below. Arrows point to
bootstrap winners in the top row or first column.

| model                 | gompertz | logistic | classical_bertalanffy | bertalanffy | bertalanffy2 | n1   | n2 |
|-----------------------|----------|----------|-----------------------|-------------|--------------|------|----|
| exponential           | ↑        | draw     | ↑                     | ↑           | draw         | draw | ←  |
| gompertz              | n/a      | draw     | draw                  | ↑           | draw         | draw | ←  |
| logistic              | draw     | n/a      | draw                  | ↑           | draw         | draw | ←  |
| classical_bertalanffy | draw     | draw     | n/a                   | ↑           | draw         | draw | ←  |
| bertalanffy           | ←        | ←        | ←                     | n/a         | ←            | draw | ←  |
| bertalanffy2          | draw     | draw     | draw                  | ↑           | n/a          | draw | ←  |
| n1                    | draw     | draw     | draw                  | draw        | draw         | n/a  | ←  |

````@julia
using Pkg
dir = @__DIR__
Pkg.activate(dir)
Pkg.instantiate()

using Random
using Statistics
using TumorGrowth
using Lux
using Plots
import PrettyPrint.pprint
using PrettyTables
using Bootstrap
using Serialization
using ProgressMeter
using .Threads
````

````
  Activating project at `~/GoogleDrive/Julia/TumorGrowth/docs/src/examples/04_model_battle`

````

## Data ingestion

Collect together all records with at least 6 measurements, from the data

````@julia
records = filter(patient_data()) do record
    record.readings >= 6
end;
````

Here's what a single record looks like:

````@julia
pprint(records[13])
````

````
@NamedTuple{Pt_hashID::String, Study_Arm::InlineStrings.String15, Study_id::Int64, Arm_id::Int64, T_weeks::Vector{Float64}, T_days::Vector{Int64}, Lesion_diam::Vector{Float64}, Lesion_vol::Vector{Float64}, Lesion_normvol::Vector{Float64}, response::InlineStrings.String7, readings::Int64}(
  Pt_hashID="d9b90f39d6a0b35cbc230adadbd50753-S1",
  Study_Arm=InlineStrings.String15("Study_1_Arm_1"),
  Study_id=1,
  Arm_id=1,
  T_weeks=[0.1, 6.0, 12.0, 18.0, 
           24.0, 36.0, 40.0, 42.0, 
           48.0],
  T_days=[-16, 39, 82, 124, 165, 
          249, 277, 292, 334],
  Lesion_diam=[17.0, 18.0, 16.0, 
               9.0, 8.0, 9.0, 7.0, 
               7.0, 7.0],
  Lesion_vol=[2554.76, 3032.64, 2129.92, 
              379.08, 266.24, 379.08, 
              178.36, 178.36, 178.36],
  Lesion_normvol=[0.000414516882387563, 
                  0.00049205423531127, 
                  0.000345585416295432, 
                  6.15067794139087e-5, 
                  4.3198177036929e-5, 
                  6.15067794139087e-5, 
                  2.89394037571615e-5, 
                  2.89394037571615e-5, 
                  2.89394037571615e-5],
  response=InlineStrings.String7("flux"),
  readings=9,
)
````

## Neural ODEs

We define some one and two-dimensional neural ODE models we want to include in our
comparison. The choice of architecture here is somewhat ad hoc and further
experimentation might give better results.

````@julia
network = Chain(
    Dense(1, 3, Lux.tanh, init_weight=Lux.zeros64),
    Dense(3, 1),
)

network2 = Chain(
    Dense(2, 2, Lux.tanh, init_weight=Lux.zeros64),
    Dense(2, 2),
)

n1 = neural(Xoshiro(123), network) # `Xoshiro` is a random number generator
n2 = neural2(Xoshiro(123), network2)
````

````
Neural2 model, (times, p) -> volumes, where length(p) = 14
  transform: log
````

## Models to be compared

````@julia
model_exs =
    [:exponential, :gompertz, :logistic, :classical_bertalanffy, :bertalanffy,
     :bertalanffy2, :n1, :n2]
models = eval.(model_exs)
````

````
8-element Vector{Any}:
 exponential (generic function with 1 method)
 gompertz (generic function with 1 method)
 logistic (generic function with 1 method)
 classical_bertalanffy (generic function with 1 method)
 bertalanffy (generic function with 1 method)
 bertalanffy2 (generic function with 1 method)
 neural (12 params)
 neural2 (14 params)
````

## Computing prediction errors on a holdout set

````@julia
holdouts = 2
errs = fill(Inf, length(records), length(models))

p = Progress(length(records))

@threads for i in eachindex(records)
    record = records[i]
    times, volumes = record.T_weeks, record.Lesion_normvol
    comparison = compare(times, volumes, models; holdouts)
    errs[i,:] = TumorGrowth.errors(comparison)
    next!(p)
end
finish!(p)
````

````
Progress: 100%|█████████████████████████████████████████████████████| Time: 3:28:43
````

````@julia
serialize(joinpath(dir, "errors.jls"), errs);
````

## Bootstrap comparisons (neural ODE's excluded)

Because the neural ODE errors contain more `NaN` values, we start with a comparison that
excludes them, discarding only those observations where `NaN` occurs in a non-neural
model.

````@julia
bad_error_rows = filter(axes(errs, 1)) do i
    es = errs[i,1:end-2]
    any(isnan, es) || any(isinf, es) || max(es...) > 0.1
end
proportion_bad = length(bad_error_rows)/size(errs, 1)
@show proportion_bad
````

````
0.0078003120124804995
````

That's less than 1%. Let's remove them:

````@julia
good_error_rows = setdiff(axes(errs, 1), bad_error_rows);
errs = errs[good_error_rows,:];
````

Errors are evidently not normally distributed (and we were not able to transform them
to approximately normal):

````@julia
plt = histogram(errs[:, 1], normalize=:pdf, alpha=0.4)
histogram!(errs[:, end-2], normalize=:pdf, alpha=0.4)
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip340">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip340)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip341">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip340)" d="M172.015 1486.45 L2352.76 1486.45 L2352.76 47.2441 L172.015 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip342">
    <rect x="172" y="47" width="2182" height="1440"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="291.959,1486.45 291.959,47.2441 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="754.067,1486.45 754.067,47.2441 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1216.17,1486.45 1216.17,47.2441 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1678.28,1486.45 1678.28,47.2441 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2140.39,1486.45 2140.39,47.2441 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,1486.45 2352.76,1486.45 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="291.959,1486.45 291.959,1467.55 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="754.067,1486.45 754.067,1467.55 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1216.17,1486.45 1216.17,1467.55 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1678.28,1486.45 1678.28,1467.55 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2140.39,1486.45 2140.39,1467.55 "/>
<path clip-path="url(#clip340)" d="M254.263 1517.37 Q250.652 1517.37 248.823 1520.93 Q247.018 1524.47 247.018 1531.6 Q247.018 1538.71 248.823 1542.27 Q250.652 1545.82 254.263 1545.82 Q257.897 1545.82 259.703 1542.27 Q261.531 1538.71 261.531 1531.6 Q261.531 1524.47 259.703 1520.93 Q257.897 1517.37 254.263 1517.37 M254.263 1513.66 Q260.073 1513.66 263.129 1518.27 Q266.207 1522.85 266.207 1531.6 Q266.207 1540.33 263.129 1544.94 Q260.073 1549.52 254.263 1549.52 Q248.453 1549.52 245.374 1544.94 Q242.319 1540.33 242.319 1531.6 Q242.319 1522.85 245.374 1518.27 Q248.453 1513.66 254.263 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M274.425 1542.97 L279.309 1542.97 L279.309 1548.85 L274.425 1548.85 L274.425 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M299.494 1517.37 Q295.883 1517.37 294.054 1520.93 Q292.249 1524.47 292.249 1531.6 Q292.249 1538.71 294.054 1542.27 Q295.883 1545.82 299.494 1545.82 Q303.128 1545.82 304.934 1542.27 Q306.763 1538.71 306.763 1531.6 Q306.763 1524.47 304.934 1520.93 Q303.128 1517.37 299.494 1517.37 M299.494 1513.66 Q305.304 1513.66 308.36 1518.27 Q311.439 1522.85 311.439 1531.6 Q311.439 1540.33 308.36 1544.94 Q305.304 1549.52 299.494 1549.52 Q293.684 1549.52 290.605 1544.94 Q287.55 1540.33 287.55 1531.6 Q287.55 1522.85 290.605 1518.27 Q293.684 1513.66 299.494 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M329.656 1517.37 Q326.045 1517.37 324.216 1520.93 Q322.411 1524.47 322.411 1531.6 Q322.411 1538.71 324.216 1542.27 Q326.045 1545.82 329.656 1545.82 Q333.29 1545.82 335.096 1542.27 Q336.924 1538.71 336.924 1531.6 Q336.924 1524.47 335.096 1520.93 Q333.29 1517.37 329.656 1517.37 M329.656 1513.66 Q335.466 1513.66 338.522 1518.27 Q341.6 1522.85 341.6 1531.6 Q341.6 1540.33 338.522 1544.94 Q335.466 1549.52 329.656 1549.52 Q323.846 1549.52 320.767 1544.94 Q317.712 1540.33 317.712 1531.6 Q317.712 1522.85 320.767 1518.27 Q323.846 1513.66 329.656 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M717.169 1517.37 Q713.558 1517.37 711.729 1520.93 Q709.924 1524.47 709.924 1531.6 Q709.924 1538.71 711.729 1542.27 Q713.558 1545.82 717.169 1545.82 Q720.803 1545.82 722.609 1542.27 Q724.438 1538.71 724.438 1531.6 Q724.438 1524.47 722.609 1520.93 Q720.803 1517.37 717.169 1517.37 M717.169 1513.66 Q722.979 1513.66 726.035 1518.27 Q729.113 1522.85 729.113 1531.6 Q729.113 1540.33 726.035 1544.94 Q722.979 1549.52 717.169 1549.52 Q711.359 1549.52 708.28 1544.94 Q705.225 1540.33 705.225 1531.6 Q705.225 1522.85 708.28 1518.27 Q711.359 1513.66 717.169 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M737.331 1542.97 L742.215 1542.97 L742.215 1548.85 L737.331 1548.85 L737.331 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M762.4 1517.37 Q758.789 1517.37 756.961 1520.93 Q755.155 1524.47 755.155 1531.6 Q755.155 1538.71 756.961 1542.27 Q758.789 1545.82 762.4 1545.82 Q766.035 1545.82 767.84 1542.27 Q769.669 1538.71 769.669 1531.6 Q769.669 1524.47 767.84 1520.93 Q766.035 1517.37 762.4 1517.37 M762.4 1513.66 Q768.21 1513.66 771.266 1518.27 Q774.345 1522.85 774.345 1531.6 Q774.345 1540.33 771.266 1544.94 Q768.21 1549.52 762.4 1549.52 Q756.59 1549.52 753.511 1544.94 Q750.456 1540.33 750.456 1531.6 Q750.456 1522.85 753.511 1518.27 Q756.59 1513.66 762.4 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M786.59 1544.91 L802.909 1544.91 L802.909 1548.85 L780.965 1548.85 L780.965 1544.91 Q783.627 1542.16 788.21 1537.53 Q792.817 1532.88 793.997 1531.53 Q796.243 1529.01 797.122 1527.27 Q798.025 1525.51 798.025 1523.82 Q798.025 1521.07 796.081 1519.33 Q794.159 1517.6 791.058 1517.6 Q788.859 1517.6 786.405 1518.36 Q783.974 1519.13 781.197 1520.68 L781.197 1515.95 Q784.021 1514.82 786.474 1514.24 Q788.928 1513.66 790.965 1513.66 Q796.335 1513.66 799.53 1516.35 Q802.724 1519.03 802.724 1523.52 Q802.724 1525.65 801.914 1527.57 Q801.127 1529.47 799.02 1532.07 Q798.442 1532.74 795.34 1535.95 Q792.238 1539.15 786.59 1544.91 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1178.23 1517.37 Q1174.62 1517.37 1172.8 1520.93 Q1170.99 1524.47 1170.99 1531.6 Q1170.99 1538.71 1172.8 1542.27 Q1174.62 1545.82 1178.23 1545.82 Q1181.87 1545.82 1183.67 1542.27 Q1185.5 1538.71 1185.5 1531.6 Q1185.5 1524.47 1183.67 1520.93 Q1181.87 1517.37 1178.23 1517.37 M1178.23 1513.66 Q1184.05 1513.66 1187.1 1518.27 Q1190.18 1522.85 1190.18 1531.6 Q1190.18 1540.33 1187.1 1544.94 Q1184.05 1549.52 1178.23 1549.52 Q1172.42 1549.52 1169.35 1544.94 Q1166.29 1540.33 1166.29 1531.6 Q1166.29 1522.85 1169.35 1518.27 Q1172.42 1513.66 1178.23 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1198.4 1542.97 L1203.28 1542.97 L1203.28 1548.85 L1198.4 1548.85 L1198.4 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1223.47 1517.37 Q1219.86 1517.37 1218.03 1520.93 Q1216.22 1524.47 1216.22 1531.6 Q1216.22 1538.71 1218.03 1542.27 Q1219.86 1545.82 1223.47 1545.82 Q1227.1 1545.82 1228.91 1542.27 Q1230.73 1538.71 1230.73 1531.6 Q1230.73 1524.47 1228.91 1520.93 Q1227.1 1517.37 1223.47 1517.37 M1223.47 1513.66 Q1229.28 1513.66 1232.33 1518.27 Q1235.41 1522.85 1235.41 1531.6 Q1235.41 1540.33 1232.33 1544.94 Q1229.28 1549.52 1223.47 1549.52 Q1217.66 1549.52 1214.58 1544.94 Q1211.52 1540.33 1211.52 1531.6 Q1211.52 1522.85 1214.58 1518.27 Q1217.66 1513.66 1223.47 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1256.48 1518.36 L1244.67 1536.81 L1256.48 1536.81 L1256.48 1518.36 M1255.25 1514.29 L1261.13 1514.29 L1261.13 1536.81 L1266.06 1536.81 L1266.06 1540.7 L1261.13 1540.7 L1261.13 1548.85 L1256.48 1548.85 L1256.48 1540.7 L1240.87 1540.7 L1240.87 1536.19 L1255.25 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1640.5 1517.37 Q1636.89 1517.37 1635.06 1520.93 Q1633.26 1524.47 1633.26 1531.6 Q1633.26 1538.71 1635.06 1542.27 Q1636.89 1545.82 1640.5 1545.82 Q1644.14 1545.82 1645.94 1542.27 Q1647.77 1538.71 1647.77 1531.6 Q1647.77 1524.47 1645.94 1520.93 Q1644.14 1517.37 1640.5 1517.37 M1640.5 1513.66 Q1646.31 1513.66 1649.37 1518.27 Q1652.45 1522.85 1652.45 1531.6 Q1652.45 1540.33 1649.37 1544.94 Q1646.31 1549.52 1640.5 1549.52 Q1634.69 1549.52 1631.62 1544.94 Q1628.56 1540.33 1628.56 1531.6 Q1628.56 1522.85 1631.62 1518.27 Q1634.69 1513.66 1640.5 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1660.67 1542.97 L1665.55 1542.97 L1665.55 1548.85 L1660.67 1548.85 L1660.67 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1685.74 1517.37 Q1682.12 1517.37 1680.3 1520.93 Q1678.49 1524.47 1678.49 1531.6 Q1678.49 1538.71 1680.3 1542.27 Q1682.12 1545.82 1685.74 1545.82 Q1689.37 1545.82 1691.18 1542.27 Q1693 1538.71 1693 1531.6 Q1693 1524.47 1691.18 1520.93 Q1689.37 1517.37 1685.74 1517.37 M1685.74 1513.66 Q1691.55 1513.66 1694.6 1518.27 Q1697.68 1522.85 1697.68 1531.6 Q1697.68 1540.33 1694.6 1544.94 Q1691.55 1549.52 1685.74 1549.52 Q1679.93 1549.52 1676.85 1544.94 Q1673.79 1540.33 1673.79 1531.6 Q1673.79 1522.85 1676.85 1518.27 Q1679.93 1513.66 1685.74 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M1716.48 1529.7 Q1713.33 1529.7 1711.48 1531.86 Q1709.65 1534.01 1709.65 1537.76 Q1709.65 1541.49 1711.48 1543.66 Q1713.33 1545.82 1716.48 1545.82 Q1719.62 1545.82 1721.45 1543.66 Q1723.31 1541.49 1723.31 1537.76 Q1723.31 1534.01 1721.45 1531.86 Q1719.62 1529.7 1716.48 1529.7 M1725.76 1515.05 L1725.76 1519.31 Q1724 1518.48 1722.19 1518.04 Q1720.41 1517.6 1718.65 1517.6 Q1714.02 1517.6 1711.57 1520.72 Q1709.14 1523.85 1708.79 1530.17 Q1710.16 1528.15 1712.22 1527.09 Q1714.28 1526 1716.75 1526 Q1721.96 1526 1724.97 1529.17 Q1728 1532.32 1728 1537.76 Q1728 1543.08 1724.86 1546.3 Q1721.71 1549.52 1716.48 1549.52 Q1710.48 1549.52 1707.31 1544.94 Q1704.14 1540.33 1704.14 1531.6 Q1704.14 1523.41 1708.03 1518.55 Q1711.92 1513.66 1718.47 1513.66 Q1720.23 1513.66 1722.01 1514.01 Q1723.81 1514.36 1725.76 1515.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2102.74 1517.37 Q2099.13 1517.37 2097.3 1520.93 Q2095.49 1524.47 2095.49 1531.6 Q2095.49 1538.71 2097.3 1542.27 Q2099.13 1545.82 2102.74 1545.82 Q2106.37 1545.82 2108.18 1542.27 Q2110.01 1538.71 2110.01 1531.6 Q2110.01 1524.47 2108.18 1520.93 Q2106.37 1517.37 2102.74 1517.37 M2102.74 1513.66 Q2108.55 1513.66 2111.61 1518.27 Q2114.68 1522.85 2114.68 1531.6 Q2114.68 1540.33 2111.61 1544.94 Q2108.55 1549.52 2102.74 1549.52 Q2096.93 1549.52 2093.85 1544.94 Q2090.8 1540.33 2090.8 1531.6 Q2090.8 1522.85 2093.85 1518.27 Q2096.93 1513.66 2102.74 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2122.9 1542.97 L2127.79 1542.97 L2127.79 1548.85 L2122.9 1548.85 L2122.9 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2147.97 1517.37 Q2144.36 1517.37 2142.53 1520.93 Q2140.73 1524.47 2140.73 1531.6 Q2140.73 1538.71 2142.53 1542.27 Q2144.36 1545.82 2147.97 1545.82 Q2151.6 1545.82 2153.41 1542.27 Q2155.24 1538.71 2155.24 1531.6 Q2155.24 1524.47 2153.41 1520.93 Q2151.6 1517.37 2147.97 1517.37 M2147.97 1513.66 Q2153.78 1513.66 2156.84 1518.27 Q2159.92 1522.85 2159.92 1531.6 Q2159.92 1540.33 2156.84 1544.94 Q2153.78 1549.52 2147.97 1549.52 Q2142.16 1549.52 2139.08 1544.94 Q2136.03 1540.33 2136.03 1531.6 Q2136.03 1522.85 2139.08 1518.27 Q2142.16 1513.66 2147.97 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2178.13 1532.44 Q2174.8 1532.44 2172.88 1534.22 Q2170.98 1536 2170.98 1539.13 Q2170.98 1542.25 2172.88 1544.03 Q2174.8 1545.82 2178.13 1545.82 Q2181.47 1545.82 2183.39 1544.03 Q2185.31 1542.23 2185.31 1539.13 Q2185.31 1536 2183.39 1534.22 Q2181.49 1532.44 2178.13 1532.44 M2173.46 1530.45 Q2170.45 1529.7 2168.76 1527.64 Q2167.09 1525.58 2167.09 1522.62 Q2167.09 1518.48 2170.03 1516.07 Q2172.99 1513.66 2178.13 1513.66 Q2183.29 1513.66 2186.23 1516.07 Q2189.17 1518.48 2189.17 1522.62 Q2189.17 1525.58 2187.48 1527.64 Q2185.82 1529.7 2182.83 1530.45 Q2186.21 1531.23 2188.09 1533.52 Q2189.98 1535.82 2189.98 1539.13 Q2189.98 1544.15 2186.91 1546.83 Q2183.85 1549.52 2178.13 1549.52 Q2172.42 1549.52 2169.34 1546.83 Q2166.28 1544.15 2166.28 1539.13 Q2166.28 1535.82 2168.18 1533.52 Q2170.08 1531.23 2173.46 1530.45 M2171.74 1523.06 Q2171.74 1525.75 2173.41 1527.25 Q2175.1 1528.76 2178.13 1528.76 Q2181.14 1528.76 2182.83 1527.25 Q2184.54 1525.75 2184.54 1523.06 Q2184.54 1520.38 2182.83 1518.87 Q2181.14 1517.37 2178.13 1517.37 Q2175.1 1517.37 2173.41 1518.87 Q2171.74 1520.38 2171.74 1523.06 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,1486.45 2352.76,1486.45 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,1249.31 2352.76,1249.31 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,1012.18 2352.76,1012.18 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,775.049 2352.76,775.049 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,537.916 2352.76,537.916 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,300.783 2352.76,300.783 "/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="172.015,63.6495 2352.76,63.6495 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,1486.45 172.015,47.2441 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,1486.45 190.912,1486.45 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,1249.31 190.912,1249.31 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,1012.18 190.912,1012.18 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,775.049 190.912,775.049 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,537.916 190.912,537.916 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,300.783 190.912,300.783 "/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="172.015,63.6495 190.912,63.6495 "/>
<path clip-path="url(#clip340)" d="M124.07 1472.25 Q120.459 1472.25 118.631 1475.81 Q116.825 1479.35 116.825 1486.48 Q116.825 1493.59 118.631 1497.15 Q120.459 1500.7 124.07 1500.7 Q127.705 1500.7 129.51 1497.15 Q131.339 1493.59 131.339 1486.48 Q131.339 1479.35 129.51 1475.81 Q127.705 1472.25 124.07 1472.25 M124.07 1468.54 Q129.881 1468.54 132.936 1473.15 Q136.015 1477.73 136.015 1486.48 Q136.015 1495.21 132.936 1499.82 Q129.881 1504.4 124.07 1504.4 Q118.26 1504.4 115.182 1499.82 Q112.126 1495.21 112.126 1486.48 Q112.126 1477.73 115.182 1473.15 Q118.26 1468.54 124.07 1468.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M54.5569 1262.66 L62.1958 1262.66 L62.1958 1236.29 L53.8856 1237.96 L53.8856 1233.7 L62.1495 1232.03 L66.8254 1232.03 L66.8254 1262.66 L74.4642 1262.66 L74.4642 1266.59 L54.5569 1266.59 L54.5569 1262.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M93.9086 1235.11 Q90.2975 1235.11 88.4688 1238.68 Q86.6632 1242.22 86.6632 1249.35 Q86.6632 1256.46 88.4688 1260.02 Q90.2975 1263.56 93.9086 1263.56 Q97.5428 1263.56 99.3483 1260.02 Q101.177 1256.46 101.177 1249.35 Q101.177 1242.22 99.3483 1238.68 Q97.5428 1235.11 93.9086 1235.11 M93.9086 1231.41 Q99.7187 1231.41 102.774 1236.02 Q105.853 1240.6 105.853 1249.35 Q105.853 1258.08 102.774 1262.68 Q99.7187 1267.27 93.9086 1267.27 Q88.0984 1267.27 85.0197 1262.68 Q81.9642 1258.08 81.9642 1249.35 Q81.9642 1240.6 85.0197 1236.02 Q88.0984 1231.41 93.9086 1231.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M124.07 1235.11 Q120.459 1235.11 118.631 1238.68 Q116.825 1242.22 116.825 1249.35 Q116.825 1256.46 118.631 1260.02 Q120.459 1263.56 124.07 1263.56 Q127.705 1263.56 129.51 1260.02 Q131.339 1256.46 131.339 1249.35 Q131.339 1242.22 129.51 1238.68 Q127.705 1235.11 124.07 1235.11 M124.07 1231.41 Q129.881 1231.41 132.936 1236.02 Q136.015 1240.6 136.015 1249.35 Q136.015 1258.08 132.936 1262.68 Q129.881 1267.27 124.07 1267.27 Q118.26 1267.27 115.182 1262.68 Q112.126 1258.08 112.126 1249.35 Q112.126 1240.6 115.182 1236.02 Q118.26 1231.41 124.07 1231.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M57.7745 1025.53 L74.0939 1025.53 L74.0939 1029.46 L52.1495 1029.46 L52.1495 1025.53 Q54.8115 1022.77 59.3949 1018.14 Q64.0013 1013.49 65.1819 1012.15 Q67.4272 1009.62 68.3068 1007.89 Q69.2096 1006.13 69.2096 1004.44 Q69.2096 1001.68 67.2652 999.948 Q65.3439 998.212 62.2421 998.212 Q60.043 998.212 57.5893 998.976 Q55.1588 999.74 52.381 1001.29 L52.381 996.568 Q55.2051 995.434 57.6588 994.855 Q60.1124 994.277 62.1495 994.277 Q67.5198 994.277 70.7142 996.962 Q73.9087 999.647 73.9087 1004.14 Q73.9087 1006.27 73.0985 1008.19 Q72.3115 1010.09 70.205 1012.68 Q69.6263 1013.35 66.5245 1016.57 Q63.4226 1019.76 57.7745 1025.53 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M93.9086 997.98 Q90.2975 997.98 88.4688 1001.55 Q86.6632 1005.09 86.6632 1012.22 Q86.6632 1019.32 88.4688 1022.89 Q90.2975 1026.43 93.9086 1026.43 Q97.5428 1026.43 99.3483 1022.89 Q101.177 1019.32 101.177 1012.22 Q101.177 1005.09 99.3483 1001.55 Q97.5428 997.98 93.9086 997.98 M93.9086 994.277 Q99.7187 994.277 102.774 998.883 Q105.853 1003.47 105.853 1012.22 Q105.853 1020.94 102.774 1025.55 Q99.7187 1030.13 93.9086 1030.13 Q88.0984 1030.13 85.0197 1025.55 Q81.9642 1020.94 81.9642 1012.22 Q81.9642 1003.47 85.0197 998.883 Q88.0984 994.277 93.9086 994.277 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M124.07 997.98 Q120.459 997.98 118.631 1001.55 Q116.825 1005.09 116.825 1012.22 Q116.825 1019.32 118.631 1022.89 Q120.459 1026.43 124.07 1026.43 Q127.705 1026.43 129.51 1022.89 Q131.339 1019.32 131.339 1012.22 Q131.339 1005.09 129.51 1001.55 Q127.705 997.98 124.07 997.98 M124.07 994.277 Q129.881 994.277 132.936 998.883 Q136.015 1003.47 136.015 1012.22 Q136.015 1020.94 132.936 1025.55 Q129.881 1030.13 124.07 1030.13 Q118.26 1030.13 115.182 1025.55 Q112.126 1020.94 112.126 1012.22 Q112.126 1003.47 115.182 998.883 Q118.26 994.277 124.07 994.277 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M67.9133 773.695 Q71.2698 774.412 73.1448 776.681 Q75.0429 778.949 75.0429 782.282 Q75.0429 787.398 71.5244 790.199 Q68.0059 793 61.5245 793 Q59.3486 793 57.0338 792.56 Q54.7421 792.144 52.2884 791.287 L52.2884 786.773 Q54.2328 787.907 56.5477 788.486 Q58.8625 789.065 61.3856 789.065 Q65.7837 789.065 68.0754 787.329 Q70.3902 785.593 70.3902 782.282 Q70.3902 779.227 68.2374 777.514 Q66.1078 775.778 62.2884 775.778 L58.2606 775.778 L58.2606 771.935 L62.4735 771.935 Q65.9226 771.935 67.7513 770.57 Q69.58 769.181 69.58 766.588 Q69.58 763.926 67.6819 762.514 Q65.8069 761.079 62.2884 761.079 Q60.3671 761.079 58.168 761.496 Q55.969 761.912 53.3301 762.792 L53.3301 758.625 Q55.9921 757.884 58.3069 757.514 Q60.6449 757.144 62.705 757.144 Q68.0291 757.144 71.1309 759.574 Q74.2327 761.982 74.2327 766.102 Q74.2327 768.972 72.5892 770.963 Q70.9457 772.931 67.9133 773.695 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M93.9086 760.847 Q90.2975 760.847 88.4688 764.412 Q86.6632 767.954 86.6632 775.083 Q86.6632 782.19 88.4688 785.755 Q90.2975 789.296 93.9086 789.296 Q97.5428 789.296 99.3483 785.755 Q101.177 782.19 101.177 775.083 Q101.177 767.954 99.3483 764.412 Q97.5428 760.847 93.9086 760.847 M93.9086 757.144 Q99.7187 757.144 102.774 761.75 Q105.853 766.333 105.853 775.083 Q105.853 783.81 102.774 788.417 Q99.7187 793 93.9086 793 Q88.0984 793 85.0197 788.417 Q81.9642 783.81 81.9642 775.083 Q81.9642 766.333 85.0197 761.75 Q88.0984 757.144 93.9086 757.144 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M124.07 760.847 Q120.459 760.847 118.631 764.412 Q116.825 767.954 116.825 775.083 Q116.825 782.19 118.631 785.755 Q120.459 789.296 124.07 789.296 Q127.705 789.296 129.51 785.755 Q131.339 782.19 131.339 775.083 Q131.339 767.954 129.51 764.412 Q127.705 760.847 124.07 760.847 M124.07 757.144 Q129.881 757.144 132.936 761.75 Q136.015 766.333 136.015 775.083 Q136.015 783.81 132.936 788.417 Q129.881 793 124.07 793 Q118.26 793 115.182 788.417 Q112.126 783.81 112.126 775.083 Q112.126 766.333 115.182 761.75 Q118.26 757.144 124.07 757.144 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M66.5939 524.71 L54.7884 543.159 L66.5939 543.159 L66.5939 524.71 M65.367 520.636 L71.2466 520.636 L71.2466 543.159 L76.1772 543.159 L76.1772 547.048 L71.2466 547.048 L71.2466 555.196 L66.5939 555.196 L66.5939 547.048 L50.9921 547.048 L50.9921 542.534 L65.367 520.636 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M93.9086 523.714 Q90.2975 523.714 88.4688 527.279 Q86.6632 530.821 86.6632 537.95 Q86.6632 545.057 88.4688 548.622 Q90.2975 552.163 93.9086 552.163 Q97.5428 552.163 99.3483 548.622 Q101.177 545.057 101.177 537.95 Q101.177 530.821 99.3483 527.279 Q97.5428 523.714 93.9086 523.714 M93.9086 520.011 Q99.7187 520.011 102.774 524.617 Q105.853 529.2 105.853 537.95 Q105.853 546.677 102.774 551.284 Q99.7187 555.867 93.9086 555.867 Q88.0984 555.867 85.0197 551.284 Q81.9642 546.677 81.9642 537.95 Q81.9642 529.2 85.0197 524.617 Q88.0984 520.011 93.9086 520.011 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M124.07 523.714 Q120.459 523.714 118.631 527.279 Q116.825 530.821 116.825 537.95 Q116.825 545.057 118.631 548.622 Q120.459 552.163 124.07 552.163 Q127.705 552.163 129.51 548.622 Q131.339 545.057 131.339 537.95 Q131.339 530.821 129.51 527.279 Q127.705 523.714 124.07 523.714 M124.07 520.011 Q129.881 520.011 132.936 524.617 Q136.015 529.2 136.015 537.95 Q136.015 546.677 132.936 551.284 Q129.881 555.867 124.07 555.867 Q118.26 555.867 115.182 551.284 Q112.126 546.677 112.126 537.95 Q112.126 529.2 115.182 524.617 Q118.26 520.011 124.07 520.011 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M53.793 283.503 L72.1494 283.503 L72.1494 287.438 L58.0754 287.438 L58.0754 295.91 Q59.0939 295.563 60.1124 295.401 Q61.131 295.215 62.1495 295.215 Q67.9365 295.215 71.3161 298.387 Q74.6957 301.558 74.6957 306.975 Q74.6957 312.553 71.2235 315.655 Q67.7513 318.734 61.4319 318.734 Q59.256 318.734 56.9875 318.364 Q54.7421 317.993 52.3347 317.252 L52.3347 312.553 Q54.418 313.688 56.6402 314.243 Q58.8625 314.799 61.3393 314.799 Q65.3439 314.799 67.6819 312.692 Q70.0198 310.586 70.0198 306.975 Q70.0198 303.364 67.6819 301.257 Q65.3439 299.151 61.3393 299.151 Q59.4643 299.151 57.5893 299.567 Q55.7375 299.984 53.793 300.864 L53.793 283.503 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M93.9086 286.581 Q90.2975 286.581 88.4688 290.146 Q86.6632 293.688 86.6632 300.817 Q86.6632 307.924 88.4688 311.489 Q90.2975 315.03 93.9086 315.03 Q97.5428 315.03 99.3483 311.489 Q101.177 307.924 101.177 300.817 Q101.177 293.688 99.3483 290.146 Q97.5428 286.581 93.9086 286.581 M93.9086 282.878 Q99.7187 282.878 102.774 287.484 Q105.853 292.067 105.853 300.817 Q105.853 309.544 102.774 314.151 Q99.7187 318.734 93.9086 318.734 Q88.0984 318.734 85.0197 314.151 Q81.9642 309.544 81.9642 300.817 Q81.9642 292.067 85.0197 287.484 Q88.0984 282.878 93.9086 282.878 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M124.07 286.581 Q120.459 286.581 118.631 290.146 Q116.825 293.688 116.825 300.817 Q116.825 307.924 118.631 311.489 Q120.459 315.03 124.07 315.03 Q127.705 315.03 129.51 311.489 Q131.339 307.924 131.339 300.817 Q131.339 293.688 129.51 290.146 Q127.705 286.581 124.07 286.581 M124.07 282.878 Q129.881 282.878 132.936 287.484 Q136.015 292.067 136.015 300.817 Q136.015 309.544 132.936 314.151 Q129.881 318.734 124.07 318.734 Q118.26 318.734 115.182 314.151 Q112.126 309.544 112.126 300.817 Q112.126 292.067 115.182 287.484 Q118.26 282.878 124.07 282.878 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M64.3254 61.7861 Q61.1773 61.7861 59.3254 63.9389 Q57.4967 66.0916 57.4967 69.8416 Q57.4967 73.5685 59.3254 75.7444 Q61.1773 77.8971 64.3254 77.8971 Q67.4735 77.8971 69.3022 75.7444 Q71.1541 73.5685 71.1541 69.8416 Q71.1541 66.0916 69.3022 63.9389 Q67.4735 61.7861 64.3254 61.7861 M73.6077 47.1334 L73.6077 51.3926 Q71.8485 50.5593 70.0429 50.1195 Q68.2606 49.6797 66.5013 49.6797 Q61.8717 49.6797 59.418 52.8047 Q56.9875 55.9297 56.6402 62.2491 Q58.006 60.2352 60.0662 59.1704 Q62.1263 58.0824 64.6032 58.0824 Q69.8115 58.0824 72.8207 61.2537 Q75.8531 64.4018 75.8531 69.8416 Q75.8531 75.1657 72.705 78.3832 Q69.5568 81.6008 64.3254 81.6008 Q58.33 81.6008 55.1588 77.0175 Q51.9875 72.4111 51.9875 63.6842 Q51.9875 55.4898 55.8764 50.6288 Q59.7652 45.7445 66.3161 45.7445 Q68.0754 45.7445 69.8578 46.0917 Q71.6633 46.439 73.6077 47.1334 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M93.9086 49.4482 Q90.2975 49.4482 88.4688 53.013 Q86.6632 56.5547 86.6632 63.6842 Q86.6632 70.7907 88.4688 74.3555 Q90.2975 77.8971 93.9086 77.8971 Q97.5428 77.8971 99.3483 74.3555 Q101.177 70.7907 101.177 63.6842 Q101.177 56.5547 99.3483 53.013 Q97.5428 49.4482 93.9086 49.4482 M93.9086 45.7445 Q99.7187 45.7445 102.774 50.351 Q105.853 54.9343 105.853 63.6842 Q105.853 72.4111 102.774 77.0175 Q99.7187 81.6008 93.9086 81.6008 Q88.0984 81.6008 85.0197 77.0175 Q81.9642 72.4111 81.9642 63.6842 Q81.9642 54.9343 85.0197 50.351 Q88.0984 45.7445 93.9086 45.7445 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M124.07 49.4482 Q120.459 49.4482 118.631 53.013 Q116.825 56.5547 116.825 63.6842 Q116.825 70.7907 118.631 74.3555 Q120.459 77.8971 124.07 77.8971 Q127.705 77.8971 129.51 74.3555 Q131.339 70.7907 131.339 63.6842 Q131.339 56.5547 129.51 53.013 Q127.705 49.4482 124.07 49.4482 M124.07 45.7445 Q129.881 45.7445 132.936 50.351 Q136.015 54.9343 136.015 63.6842 Q136.015 72.4111 132.936 77.0175 Q129.881 81.6008 124.07 81.6008 Q118.26 81.6008 115.182 77.0175 Q112.126 72.4111 112.126 63.6842 Q112.126 54.9343 115.182 50.351 Q118.26 45.7445 124.07 45.7445 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip342)" d="M291.959 47.2441 L291.959 1486.45 L315.065 1486.45 L315.065 47.2441 L291.959 47.2441 L291.959 47.2441  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="291.959,47.2441 291.959,1486.45 315.065,1486.45 315.065,47.2441 291.959,47.2441 "/>
<path clip-path="url(#clip342)" d="M315.065 1229.18 L315.065 1486.45 L338.17 1486.45 L338.17 1229.18 L315.065 1229.18 L315.065 1229.18  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="315.065,1229.18 315.065,1486.45 338.17,1486.45 338.17,1229.18 315.065,1229.18 "/>
<path clip-path="url(#clip342)" d="M338.17 1378.32 L338.17 1486.45 L361.276 1486.45 L361.276 1378.32 L338.17 1378.32 L338.17 1378.32  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="338.17,1378.32 338.17,1486.45 361.276,1486.45 361.276,1378.32 338.17,1378.32 "/>
<path clip-path="url(#clip342)" d="M361.276 1389.51 L361.276 1486.45 L384.381 1486.45 L384.381 1389.51 L361.276 1389.51 L361.276 1389.51  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="361.276,1389.51 361.276,1486.45 384.381,1486.45 384.381,1389.51 361.276,1389.51 "/>
<path clip-path="url(#clip342)" d="M384.381 1441.71 L384.381 1486.45 L407.486 1486.45 L407.486 1441.71 L384.381 1441.71 L384.381 1441.71  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="384.381,1441.71 384.381,1486.45 407.486,1486.45 407.486,1441.71 384.381,1441.71 "/>
<path clip-path="url(#clip342)" d="M407.486 1434.25 L407.486 1486.45 L430.592 1486.45 L430.592 1434.25 L407.486 1434.25 L407.486 1434.25  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="407.486,1434.25 407.486,1486.45 430.592,1486.45 430.592,1434.25 407.486,1434.25 "/>
<path clip-path="url(#clip342)" d="M430.592 1441.71 L430.592 1486.45 L453.697 1486.45 L453.697 1441.71 L430.592 1441.71 L430.592 1441.71  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="430.592,1441.71 430.592,1486.45 453.697,1486.45 453.697,1441.71 430.592,1441.71 "/>
<path clip-path="url(#clip342)" d="M453.697 1423.06 L453.697 1486.45 L476.802 1486.45 L476.802 1423.06 L453.697 1423.06 L453.697 1423.06  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="453.697,1423.06 453.697,1486.45 476.802,1486.45 476.802,1423.06 453.697,1423.06 "/>
<path clip-path="url(#clip342)" d="M476.802 1456.62 L476.802 1486.45 L499.908 1486.45 L499.908 1456.62 L476.802 1456.62 L476.802 1456.62  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="476.802,1456.62 476.802,1486.45 499.908,1486.45 499.908,1456.62 476.802,1456.62 "/>
<path clip-path="url(#clip342)" d="M499.908 1452.89 L499.908 1486.45 L523.013 1486.45 L523.013 1452.89 L499.908 1452.89 L499.908 1452.89  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="499.908,1452.89 499.908,1486.45 523.013,1486.45 523.013,1452.89 499.908,1452.89 "/>
<path clip-path="url(#clip342)" d="M523.013 1475.26 L523.013 1486.45 L546.119 1486.45 L546.119 1475.26 L523.013 1475.26 L523.013 1475.26  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="523.013,1475.26 523.013,1486.45 546.119,1486.45 546.119,1475.26 523.013,1475.26 "/>
<path clip-path="url(#clip342)" d="M546.119 1467.81 L546.119 1486.45 L569.224 1486.45 L569.224 1467.81 L546.119 1467.81 L546.119 1467.81  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="546.119,1467.81 546.119,1486.45 569.224,1486.45 569.224,1467.81 546.119,1467.81 "/>
<path clip-path="url(#clip342)" d="M569.224 1460.35 L569.224 1486.45 L592.329 1486.45 L592.329 1460.35 L569.224 1460.35 L569.224 1460.35  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="569.224,1460.35 569.224,1486.45 592.329,1486.45 592.329,1460.35 569.224,1460.35 "/>
<path clip-path="url(#clip342)" d="M592.329 1478.99 L592.329 1486.45 L615.435 1486.45 L615.435 1478.99 L592.329 1478.99 L592.329 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="592.329,1478.99 592.329,1486.45 615.435,1486.45 615.435,1478.99 592.329,1478.99 "/>
<path clip-path="url(#clip342)" d="M615.435 1478.99 L615.435 1486.45 L638.54 1486.45 L638.54 1478.99 L615.435 1478.99 L615.435 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="615.435,1478.99 615.435,1486.45 638.54,1486.45 638.54,1478.99 615.435,1478.99 "/>
<path clip-path="url(#clip342)" d="M638.54 1478.99 L638.54 1486.45 L661.646 1486.45 L661.646 1478.99 L638.54 1478.99 L638.54 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="638.54,1478.99 638.54,1486.45 661.646,1486.45 661.646,1478.99 638.54,1478.99 "/>
<path clip-path="url(#clip342)" d="M661.646 1486.45 L661.646 1486.45 L684.751 1486.45 L684.751 1486.45 L661.646 1486.45 L661.646 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="661.646,1486.45 661.646,1486.45 684.751,1486.45 661.646,1486.45 "/>
<path clip-path="url(#clip342)" d="M684.751 1471.53 L684.751 1486.45 L707.856 1486.45 L707.856 1471.53 L684.751 1471.53 L684.751 1471.53  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="684.751,1471.53 684.751,1486.45 707.856,1486.45 707.856,1471.53 684.751,1471.53 "/>
<path clip-path="url(#clip342)" d="M707.856 1482.72 L707.856 1486.45 L730.962 1486.45 L730.962 1482.72 L707.856 1482.72 L707.856 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="707.856,1482.72 707.856,1486.45 730.962,1486.45 730.962,1482.72 707.856,1482.72 "/>
<path clip-path="url(#clip342)" d="M730.962 1482.72 L730.962 1486.45 L754.067 1486.45 L754.067 1482.72 L730.962 1482.72 L730.962 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="730.962,1482.72 730.962,1486.45 754.067,1486.45 754.067,1482.72 730.962,1482.72 "/>
<path clip-path="url(#clip342)" d="M754.067 1471.53 L754.067 1486.45 L777.172 1486.45 L777.172 1471.53 L754.067 1471.53 L754.067 1471.53  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="754.067,1471.53 754.067,1486.45 777.172,1486.45 777.172,1471.53 754.067,1471.53 "/>
<path clip-path="url(#clip342)" d="M777.172 1478.99 L777.172 1486.45 L800.278 1486.45 L800.278 1478.99 L777.172 1478.99 L777.172 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="777.172,1478.99 777.172,1486.45 800.278,1486.45 800.278,1478.99 777.172,1478.99 "/>
<path clip-path="url(#clip342)" d="M800.278 1478.99 L800.278 1486.45 L823.383 1486.45 L823.383 1478.99 L800.278 1478.99 L800.278 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="800.278,1478.99 800.278,1486.45 823.383,1486.45 823.383,1478.99 800.278,1478.99 "/>
<path clip-path="url(#clip342)" d="M823.383 1482.72 L823.383 1486.45 L846.489 1486.45 L846.489 1482.72 L823.383 1482.72 L823.383 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="823.383,1482.72 823.383,1486.45 846.489,1486.45 846.489,1482.72 823.383,1482.72 "/>
<path clip-path="url(#clip342)" d="M846.489 1482.72 L846.489 1486.45 L869.594 1486.45 L869.594 1482.72 L846.489 1482.72 L846.489 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="846.489,1482.72 846.489,1486.45 869.594,1486.45 869.594,1482.72 846.489,1482.72 "/>
<path clip-path="url(#clip342)" d="M869.594 1478.99 L869.594 1486.45 L892.699 1486.45 L892.699 1478.99 L869.594 1478.99 L869.594 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="869.594,1478.99 869.594,1486.45 892.699,1486.45 892.699,1478.99 869.594,1478.99 "/>
<path clip-path="url(#clip342)" d="M892.699 1486.45 L892.699 1486.45 L915.805 1486.45 L915.805 1486.45 L892.699 1486.45 L892.699 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="892.699,1486.45 892.699,1486.45 915.805,1486.45 892.699,1486.45 "/>
<path clip-path="url(#clip342)" d="M915.805 1486.45 L915.805 1486.45 L938.91 1486.45 L938.91 1486.45 L915.805 1486.45 L915.805 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="915.805,1486.45 915.805,1486.45 938.91,1486.45 915.805,1486.45 "/>
<path clip-path="url(#clip342)" d="M938.91 1482.72 L938.91 1486.45 L962.015 1486.45 L962.015 1482.72 L938.91 1482.72 L938.91 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="938.91,1482.72 938.91,1486.45 962.015,1486.45 962.015,1482.72 938.91,1482.72 "/>
<path clip-path="url(#clip342)" d="M962.015 1486.45 L962.015 1486.45 L985.121 1486.45 L985.121 1486.45 L962.015 1486.45 L962.015 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="962.015,1486.45 962.015,1486.45 985.121,1486.45 962.015,1486.45 "/>
<path clip-path="url(#clip342)" d="M985.121 1478.99 L985.121 1486.45 L1008.23 1486.45 L1008.23 1478.99 L985.121 1478.99 L985.121 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="985.121,1478.99 985.121,1486.45 1008.23,1486.45 1008.23,1478.99 985.121,1478.99 "/>
<path clip-path="url(#clip342)" d="M1008.23 1486.45 L1008.23 1486.45 L1031.33 1486.45 L1031.33 1486.45 L1008.23 1486.45 L1008.23 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1008.23,1486.45 1008.23,1486.45 1031.33,1486.45 1008.23,1486.45 "/>
<path clip-path="url(#clip342)" d="M1031.33 1482.72 L1031.33 1486.45 L1054.44 1486.45 L1054.44 1482.72 L1031.33 1482.72 L1031.33 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1031.33,1482.72 1031.33,1486.45 1054.44,1486.45 1054.44,1482.72 1031.33,1482.72 "/>
<path clip-path="url(#clip342)" d="M1054.44 1486.45 L1054.44 1486.45 L1077.54 1486.45 L1077.54 1486.45 L1054.44 1486.45 L1054.44 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1054.44,1486.45 1054.44,1486.45 1077.54,1486.45 1054.44,1486.45 "/>
<path clip-path="url(#clip342)" d="M1077.54 1482.72 L1077.54 1486.45 L1100.65 1486.45 L1100.65 1482.72 L1077.54 1482.72 L1077.54 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1077.54,1482.72 1077.54,1486.45 1100.65,1486.45 1100.65,1482.72 1077.54,1482.72 "/>
<path clip-path="url(#clip342)" d="M1100.65 1482.72 L1100.65 1486.45 L1123.75 1486.45 L1123.75 1482.72 L1100.65 1482.72 L1100.65 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1100.65,1482.72 1100.65,1486.45 1123.75,1486.45 1123.75,1482.72 1100.65,1482.72 "/>
<path clip-path="url(#clip342)" d="M1123.75 1482.72 L1123.75 1486.45 L1146.86 1486.45 L1146.86 1482.72 L1123.75 1482.72 L1123.75 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1123.75,1482.72 1123.75,1486.45 1146.86,1486.45 1146.86,1482.72 1123.75,1482.72 "/>
<path clip-path="url(#clip342)" d="M1146.86 1486.45 L1146.86 1486.45 L1169.96 1486.45 L1169.96 1486.45 L1146.86 1486.45 L1146.86 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1146.86,1486.45 1146.86,1486.45 1169.96,1486.45 1146.86,1486.45 "/>
<path clip-path="url(#clip342)" d="M1169.96 1486.45 L1169.96 1486.45 L1193.07 1486.45 L1193.07 1486.45 L1169.96 1486.45 L1169.96 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1169.96,1486.45 1169.96,1486.45 1193.07,1486.45 1169.96,1486.45 "/>
<path clip-path="url(#clip342)" d="M1193.07 1486.45 L1193.07 1486.45 L1216.17 1486.45 L1216.17 1486.45 L1193.07 1486.45 L1193.07 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1193.07,1486.45 1193.07,1486.45 1216.17,1486.45 1193.07,1486.45 "/>
<path clip-path="url(#clip342)" d="M1216.17 1478.99 L1216.17 1486.45 L1239.28 1486.45 L1239.28 1478.99 L1216.17 1478.99 L1216.17 1478.99  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1216.17,1478.99 1216.17,1486.45 1239.28,1486.45 1239.28,1478.99 1216.17,1478.99 "/>
<path clip-path="url(#clip342)" d="M1239.28 1486.45 L1239.28 1486.45 L1262.39 1486.45 L1262.39 1486.45 L1239.28 1486.45 L1239.28 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1239.28,1486.45 1239.28,1486.45 1262.39,1486.45 1239.28,1486.45 "/>
<path clip-path="url(#clip342)" d="M1262.39 1482.72 L1262.39 1486.45 L1285.49 1486.45 L1285.49 1482.72 L1262.39 1482.72 L1262.39 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1262.39,1482.72 1262.39,1486.45 1285.49,1486.45 1285.49,1482.72 1262.39,1482.72 "/>
<path clip-path="url(#clip342)" d="M1285.49 1486.45 L1285.49 1486.45 L1308.6 1486.45 L1308.6 1486.45 L1285.49 1486.45 L1285.49 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1285.49,1486.45 1285.49,1486.45 1308.6,1486.45 1285.49,1486.45 "/>
<path clip-path="url(#clip342)" d="M1308.6 1486.45 L1308.6 1486.45 L1331.7 1486.45 L1331.7 1486.45 L1308.6 1486.45 L1308.6 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1308.6,1486.45 1308.6,1486.45 1331.7,1486.45 1308.6,1486.45 "/>
<path clip-path="url(#clip342)" d="M1331.7 1482.72 L1331.7 1486.45 L1354.81 1486.45 L1354.81 1482.72 L1331.7 1482.72 L1331.7 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1331.7,1482.72 1331.7,1486.45 1354.81,1486.45 1354.81,1482.72 1331.7,1482.72 "/>
<path clip-path="url(#clip342)" d="M1354.81 1486.45 L1354.81 1486.45 L1377.91 1486.45 L1377.91 1486.45 L1354.81 1486.45 L1354.81 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1354.81,1486.45 1354.81,1486.45 1377.91,1486.45 1354.81,1486.45 "/>
<path clip-path="url(#clip342)" d="M1377.91 1486.45 L1377.91 1486.45 L1401.02 1486.45 L1401.02 1486.45 L1377.91 1486.45 L1377.91 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1377.91,1486.45 1377.91,1486.45 1401.02,1486.45 1377.91,1486.45 "/>
<path clip-path="url(#clip342)" d="M1401.02 1486.45 L1401.02 1486.45 L1424.12 1486.45 L1424.12 1486.45 L1401.02 1486.45 L1401.02 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1401.02,1486.45 1401.02,1486.45 1424.12,1486.45 1401.02,1486.45 "/>
<path clip-path="url(#clip342)" d="M1424.12 1486.45 L1424.12 1486.45 L1447.23 1486.45 L1447.23 1486.45 L1424.12 1486.45 L1424.12 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1424.12,1486.45 1424.12,1486.45 1447.23,1486.45 1424.12,1486.45 "/>
<path clip-path="url(#clip342)" d="M1447.23 1482.72 L1447.23 1486.45 L1470.33 1486.45 L1470.33 1482.72 L1447.23 1482.72 L1447.23 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1447.23,1482.72 1447.23,1486.45 1470.33,1486.45 1470.33,1482.72 1447.23,1482.72 "/>
<path clip-path="url(#clip342)" d="M1470.33 1486.45 L1470.33 1486.45 L1493.44 1486.45 L1493.44 1486.45 L1470.33 1486.45 L1470.33 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1470.33,1486.45 1470.33,1486.45 1493.44,1486.45 1470.33,1486.45 "/>
<path clip-path="url(#clip342)" d="M1493.44 1486.45 L1493.44 1486.45 L1516.54 1486.45 L1516.54 1486.45 L1493.44 1486.45 L1493.44 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1493.44,1486.45 1493.44,1486.45 1516.54,1486.45 1493.44,1486.45 "/>
<path clip-path="url(#clip342)" d="M1516.54 1482.72 L1516.54 1486.45 L1539.65 1486.45 L1539.65 1482.72 L1516.54 1482.72 L1516.54 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1516.54,1482.72 1516.54,1486.45 1539.65,1486.45 1539.65,1482.72 1516.54,1482.72 "/>
<path clip-path="url(#clip342)" d="M1539.65 1486.45 L1539.65 1486.45 L1562.76 1486.45 L1562.76 1486.45 L1539.65 1486.45 L1539.65 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1539.65,1486.45 1539.65,1486.45 1562.76,1486.45 1539.65,1486.45 "/>
<path clip-path="url(#clip342)" d="M1562.76 1486.45 L1562.76 1486.45 L1585.86 1486.45 L1585.86 1486.45 L1562.76 1486.45 L1562.76 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1562.76,1486.45 1562.76,1486.45 1585.86,1486.45 1562.76,1486.45 "/>
<path clip-path="url(#clip342)" d="M1585.86 1486.45 L1585.86 1486.45 L1608.97 1486.45 L1608.97 1486.45 L1585.86 1486.45 L1585.86 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1585.86,1486.45 1585.86,1486.45 1608.97,1486.45 1585.86,1486.45 "/>
<path clip-path="url(#clip342)" d="M1608.97 1486.45 L1608.97 1486.45 L1632.07 1486.45 L1632.07 1486.45 L1608.97 1486.45 L1608.97 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1608.97,1486.45 1608.97,1486.45 1632.07,1486.45 1608.97,1486.45 "/>
<path clip-path="url(#clip342)" d="M1632.07 1486.45 L1632.07 1486.45 L1655.18 1486.45 L1655.18 1486.45 L1632.07 1486.45 L1632.07 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1632.07,1486.45 1632.07,1486.45 1655.18,1486.45 1632.07,1486.45 "/>
<path clip-path="url(#clip342)" d="M1655.18 1486.45 L1655.18 1486.45 L1678.28 1486.45 L1678.28 1486.45 L1655.18 1486.45 L1655.18 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1655.18,1486.45 1655.18,1486.45 1678.28,1486.45 1655.18,1486.45 "/>
<path clip-path="url(#clip342)" d="M1678.28 1486.45 L1678.28 1486.45 L1701.39 1486.45 L1701.39 1486.45 L1678.28 1486.45 L1678.28 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1678.28,1486.45 1678.28,1486.45 1701.39,1486.45 1678.28,1486.45 "/>
<path clip-path="url(#clip342)" d="M1701.39 1486.45 L1701.39 1486.45 L1724.49 1486.45 L1724.49 1486.45 L1701.39 1486.45 L1701.39 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1701.39,1486.45 1701.39,1486.45 1724.49,1486.45 1701.39,1486.45 "/>
<path clip-path="url(#clip342)" d="M1724.49 1486.45 L1724.49 1486.45 L1747.6 1486.45 L1747.6 1486.45 L1724.49 1486.45 L1724.49 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1724.49,1486.45 1724.49,1486.45 1747.6,1486.45 1724.49,1486.45 "/>
<path clip-path="url(#clip342)" d="M1747.6 1486.45 L1747.6 1486.45 L1770.7 1486.45 L1770.7 1486.45 L1747.6 1486.45 L1747.6 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1747.6,1486.45 1747.6,1486.45 1770.7,1486.45 1747.6,1486.45 "/>
<path clip-path="url(#clip342)" d="M1770.7 1486.45 L1770.7 1486.45 L1793.81 1486.45 L1793.81 1486.45 L1770.7 1486.45 L1770.7 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1770.7,1486.45 1770.7,1486.45 1793.81,1486.45 1770.7,1486.45 "/>
<path clip-path="url(#clip342)" d="M1793.81 1486.45 L1793.81 1486.45 L1816.91 1486.45 L1816.91 1486.45 L1793.81 1486.45 L1793.81 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1793.81,1486.45 1793.81,1486.45 1816.91,1486.45 1793.81,1486.45 "/>
<path clip-path="url(#clip342)" d="M1816.91 1486.45 L1816.91 1486.45 L1840.02 1486.45 L1840.02 1486.45 L1816.91 1486.45 L1816.91 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1816.91,1486.45 1816.91,1486.45 1840.02,1486.45 1816.91,1486.45 "/>
<path clip-path="url(#clip342)" d="M1840.02 1486.45 L1840.02 1486.45 L1863.13 1486.45 L1863.13 1486.45 L1840.02 1486.45 L1840.02 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1840.02,1486.45 1840.02,1486.45 1863.13,1486.45 1840.02,1486.45 "/>
<path clip-path="url(#clip342)" d="M1863.13 1486.45 L1863.13 1486.45 L1886.23 1486.45 L1886.23 1486.45 L1863.13 1486.45 L1863.13 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1863.13,1486.45 1863.13,1486.45 1886.23,1486.45 1863.13,1486.45 "/>
<path clip-path="url(#clip342)" d="M1886.23 1486.45 L1886.23 1486.45 L1909.34 1486.45 L1909.34 1486.45 L1886.23 1486.45 L1886.23 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1886.23,1486.45 1886.23,1486.45 1909.34,1486.45 1886.23,1486.45 "/>
<path clip-path="url(#clip342)" d="M1909.34 1486.45 L1909.34 1486.45 L1932.44 1486.45 L1932.44 1486.45 L1909.34 1486.45 L1909.34 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1909.34,1486.45 1909.34,1486.45 1932.44,1486.45 1909.34,1486.45 "/>
<path clip-path="url(#clip342)" d="M1932.44 1486.45 L1932.44 1486.45 L1955.55 1486.45 L1955.55 1486.45 L1932.44 1486.45 L1932.44 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1932.44,1486.45 1932.44,1486.45 1955.55,1486.45 1932.44,1486.45 "/>
<path clip-path="url(#clip342)" d="M1955.55 1486.45 L1955.55 1486.45 L1978.65 1486.45 L1978.65 1486.45 L1955.55 1486.45 L1955.55 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1955.55,1486.45 1955.55,1486.45 1978.65,1486.45 1955.55,1486.45 "/>
<path clip-path="url(#clip342)" d="M1978.65 1486.45 L1978.65 1486.45 L2001.76 1486.45 L2001.76 1486.45 L1978.65 1486.45 L1978.65 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1978.65,1486.45 1978.65,1486.45 2001.76,1486.45 1978.65,1486.45 "/>
<path clip-path="url(#clip342)" d="M2001.76 1486.45 L2001.76 1486.45 L2024.86 1486.45 L2024.86 1486.45 L2001.76 1486.45 L2001.76 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2001.76,1486.45 2001.76,1486.45 2024.86,1486.45 2001.76,1486.45 "/>
<path clip-path="url(#clip342)" d="M2024.86 1486.45 L2024.86 1486.45 L2047.97 1486.45 L2047.97 1486.45 L2024.86 1486.45 L2024.86 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2024.86,1486.45 2024.86,1486.45 2047.97,1486.45 2024.86,1486.45 "/>
<path clip-path="url(#clip342)" d="M2047.97 1486.45 L2047.97 1486.45 L2071.07 1486.45 L2071.07 1486.45 L2047.97 1486.45 L2047.97 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2047.97,1486.45 2047.97,1486.45 2071.07,1486.45 2047.97,1486.45 "/>
<path clip-path="url(#clip342)" d="M2071.07 1486.45 L2071.07 1486.45 L2094.18 1486.45 L2094.18 1486.45 L2071.07 1486.45 L2071.07 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2071.07,1486.45 2071.07,1486.45 2094.18,1486.45 2071.07,1486.45 "/>
<path clip-path="url(#clip342)" d="M2094.18 1486.45 L2094.18 1486.45 L2117.28 1486.45 L2117.28 1486.45 L2094.18 1486.45 L2094.18 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2094.18,1486.45 2094.18,1486.45 2117.28,1486.45 2094.18,1486.45 "/>
<path clip-path="url(#clip342)" d="M2117.28 1486.45 L2117.28 1486.45 L2140.39 1486.45 L2140.39 1486.45 L2117.28 1486.45 L2117.28 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2117.28,1486.45 2117.28,1486.45 2140.39,1486.45 2117.28,1486.45 "/>
<path clip-path="url(#clip342)" d="M2140.39 1482.72 L2140.39 1486.45 L2163.5 1486.45 L2163.5 1482.72 L2140.39 1482.72 L2140.39 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2140.39,1482.72 2140.39,1486.45 2163.5,1486.45 2163.5,1482.72 2140.39,1482.72 "/>
<path clip-path="url(#clip342)" d="M2163.5 1486.45 L2163.5 1486.45 L2186.6 1486.45 L2186.6 1486.45 L2163.5 1486.45 L2163.5 1486.45  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2163.5,1486.45 2163.5,1486.45 2186.6,1486.45 2163.5,1486.45 "/>
<path clip-path="url(#clip342)" d="M2186.6 1482.72 L2186.6 1486.45 L2209.71 1486.45 L2209.71 1482.72 L2186.6 1482.72 L2186.6 1482.72  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2186.6,1482.72 2186.6,1486.45 2209.71,1486.45 2209.71,1482.72 2186.6,1482.72 "/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="303.512" cy="47.2441" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="326.618" cy="1229.18" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="349.723" cy="1378.32" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="372.828" cy="1389.51" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="395.934" cy="1441.71" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="419.039" cy="1434.25" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="442.144" cy="1441.71" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="465.25" cy="1423.06" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="488.355" cy="1456.62" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="511.461" cy="1452.89" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="534.566" cy="1475.26" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="557.671" cy="1467.81" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="580.777" cy="1460.35" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="603.882" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="626.987" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="650.093" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="673.198" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="696.304" cy="1471.53" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="719.409" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="742.514" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="765.62" cy="1471.53" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="788.725" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="811.83" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="834.936" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="858.041" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="881.147" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="904.252" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="927.357" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="950.463" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="973.568" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="996.674" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1019.78" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1042.88" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1065.99" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1089.1" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1112.2" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1135.31" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1158.41" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1181.52" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1204.62" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1227.73" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1250.83" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1273.94" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1297.04" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1320.15" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1343.25" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1366.36" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1389.46" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1412.57" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1435.68" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1458.78" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1481.89" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1504.99" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1528.1" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1551.2" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1574.31" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1597.41" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1620.52" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1643.62" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1666.73" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1689.83" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1712.94" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1736.05" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1759.15" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1782.26" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1805.36" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1828.47" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1851.57" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1874.68" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1897.78" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1920.89" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1943.99" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1967.1" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="1990.2" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2013.31" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2036.42" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2059.52" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2082.63" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2105.73" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2128.84" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2151.94" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2175.05" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#009af9; stroke:none; fill-opacity:0" cx="2198.15" cy="1482.72" r="2"/>
<path clip-path="url(#clip342)" d="M291.959 65.8866 L291.959 1486.45 L315.065 1486.45 L315.065 65.8866 L291.959 65.8866 L291.959 65.8866  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="291.959,65.8866 291.959,1486.45 315.065,1486.45 315.065,65.8866 291.959,65.8866 "/>
<path clip-path="url(#clip342)" d="M315.065 1218 L315.065 1486.45 L338.17 1486.45 L338.17 1218 L315.065 1218 L315.065 1218  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="315.065,1218 315.065,1486.45 338.17,1486.45 338.17,1218 315.065,1218 "/>
<path clip-path="url(#clip342)" d="M338.17 1355.95 L338.17 1486.45 L361.276 1486.45 L361.276 1355.95 L338.17 1355.95 L338.17 1355.95  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="338.17,1355.95 338.17,1486.45 361.276,1486.45 361.276,1355.95 338.17,1355.95 "/>
<path clip-path="url(#clip342)" d="M361.276 1404.42 L361.276 1486.45 L384.381 1486.45 L384.381 1404.42 L361.276 1404.42 L361.276 1404.42  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="361.276,1404.42 361.276,1486.45 384.381,1486.45 384.381,1404.42 361.276,1404.42 "/>
<path clip-path="url(#clip342)" d="M384.381 1393.24 L384.381 1486.45 L407.486 1486.45 L407.486 1393.24 L384.381 1393.24 L384.381 1393.24  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="384.381,1393.24 384.381,1486.45 407.486,1486.45 407.486,1393.24 384.381,1393.24 "/>
<path clip-path="url(#clip342)" d="M407.486 1449.16 L407.486 1486.45 L430.592 1486.45 L430.592 1449.16 L407.486 1449.16 L407.486 1449.16  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="407.486,1449.16 407.486,1486.45 430.592,1486.45 430.592,1449.16 407.486,1449.16 "/>
<path clip-path="url(#clip342)" d="M430.592 1449.16 L430.592 1486.45 L453.697 1486.45 L453.697 1449.16 L430.592 1449.16 L430.592 1449.16  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="430.592,1449.16 430.592,1486.45 453.697,1486.45 453.697,1449.16 430.592,1449.16 "/>
<path clip-path="url(#clip342)" d="M453.697 1464.08 L453.697 1486.45 L476.802 1486.45 L476.802 1464.08 L453.697 1464.08 L453.697 1464.08  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="453.697,1464.08 453.697,1486.45 476.802,1486.45 476.802,1464.08 453.697,1464.08 "/>
<path clip-path="url(#clip342)" d="M476.802 1441.71 L476.802 1486.45 L499.908 1486.45 L499.908 1441.71 L476.802 1441.71 L476.802 1441.71  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="476.802,1441.71 476.802,1486.45 499.908,1486.45 499.908,1441.71 476.802,1441.71 "/>
<path clip-path="url(#clip342)" d="M499.908 1460.35 L499.908 1486.45 L523.013 1486.45 L523.013 1460.35 L499.908 1460.35 L499.908 1460.35  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="499.908,1460.35 499.908,1486.45 523.013,1486.45 523.013,1460.35 499.908,1460.35 "/>
<path clip-path="url(#clip342)" d="M523.013 1478.99 L523.013 1486.45 L546.119 1486.45 L546.119 1478.99 L523.013 1478.99 L523.013 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="523.013,1478.99 523.013,1486.45 546.119,1486.45 546.119,1478.99 523.013,1478.99 "/>
<path clip-path="url(#clip342)" d="M546.119 1471.53 L546.119 1486.45 L569.224 1486.45 L569.224 1471.53 L546.119 1471.53 L546.119 1471.53  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="546.119,1471.53 546.119,1486.45 569.224,1486.45 569.224,1471.53 546.119,1471.53 "/>
<path clip-path="url(#clip342)" d="M569.224 1467.81 L569.224 1486.45 L592.329 1486.45 L592.329 1467.81 L569.224 1467.81 L569.224 1467.81  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="569.224,1467.81 569.224,1486.45 592.329,1486.45 592.329,1467.81 569.224,1467.81 "/>
<path clip-path="url(#clip342)" d="M592.329 1464.08 L592.329 1486.45 L615.435 1486.45 L615.435 1464.08 L592.329 1464.08 L592.329 1464.08  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="592.329,1464.08 592.329,1486.45 615.435,1486.45 615.435,1464.08 592.329,1464.08 "/>
<path clip-path="url(#clip342)" d="M615.435 1478.99 L615.435 1486.45 L638.54 1486.45 L638.54 1478.99 L615.435 1478.99 L615.435 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="615.435,1478.99 615.435,1486.45 638.54,1486.45 638.54,1478.99 615.435,1478.99 "/>
<path clip-path="url(#clip342)" d="M638.54 1475.26 L638.54 1486.45 L661.646 1486.45 L661.646 1475.26 L638.54 1475.26 L638.54 1475.26  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="638.54,1475.26 638.54,1486.45 661.646,1486.45 661.646,1475.26 638.54,1475.26 "/>
<path clip-path="url(#clip342)" d="M661.646 1475.26 L661.646 1486.45 L684.751 1486.45 L684.751 1475.26 L661.646 1475.26 L661.646 1475.26  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="661.646,1475.26 661.646,1486.45 684.751,1486.45 684.751,1475.26 661.646,1475.26 "/>
<path clip-path="url(#clip342)" d="M684.751 1478.99 L684.751 1486.45 L707.856 1486.45 L707.856 1478.99 L684.751 1478.99 L684.751 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="684.751,1478.99 684.751,1486.45 707.856,1486.45 707.856,1478.99 684.751,1478.99 "/>
<path clip-path="url(#clip342)" d="M707.856 1482.72 L707.856 1486.45 L730.962 1486.45 L730.962 1482.72 L707.856 1482.72 L707.856 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="707.856,1482.72 707.856,1486.45 730.962,1486.45 730.962,1482.72 707.856,1482.72 "/>
<path clip-path="url(#clip342)" d="M730.962 1482.72 L730.962 1486.45 L754.067 1486.45 L754.067 1482.72 L730.962 1482.72 L730.962 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="730.962,1482.72 730.962,1486.45 754.067,1486.45 754.067,1482.72 730.962,1482.72 "/>
<path clip-path="url(#clip342)" d="M754.067 1478.99 L754.067 1486.45 L777.172 1486.45 L777.172 1478.99 L754.067 1478.99 L754.067 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="754.067,1478.99 754.067,1486.45 777.172,1486.45 777.172,1478.99 754.067,1478.99 "/>
<path clip-path="url(#clip342)" d="M777.172 1482.72 L777.172 1486.45 L800.278 1486.45 L800.278 1482.72 L777.172 1482.72 L777.172 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="777.172,1482.72 777.172,1486.45 800.278,1486.45 800.278,1482.72 777.172,1482.72 "/>
<path clip-path="url(#clip342)" d="M800.278 1478.99 L800.278 1486.45 L823.383 1486.45 L823.383 1478.99 L800.278 1478.99 L800.278 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="800.278,1478.99 800.278,1486.45 823.383,1486.45 823.383,1478.99 800.278,1478.99 "/>
<path clip-path="url(#clip342)" d="M823.383 1482.72 L823.383 1486.45 L846.489 1486.45 L846.489 1482.72 L823.383 1482.72 L823.383 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="823.383,1482.72 823.383,1486.45 846.489,1486.45 846.489,1482.72 823.383,1482.72 "/>
<path clip-path="url(#clip342)" d="M846.489 1486.45 L846.489 1486.45 L869.594 1486.45 L869.594 1486.45 L846.489 1486.45 L846.489 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="846.489,1486.45 846.489,1486.45 869.594,1486.45 846.489,1486.45 "/>
<path clip-path="url(#clip342)" d="M869.594 1482.72 L869.594 1486.45 L892.699 1486.45 L892.699 1482.72 L869.594 1482.72 L869.594 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="869.594,1482.72 869.594,1486.45 892.699,1486.45 892.699,1482.72 869.594,1482.72 "/>
<path clip-path="url(#clip342)" d="M892.699 1486.45 L892.699 1486.45 L915.805 1486.45 L915.805 1486.45 L892.699 1486.45 L892.699 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="892.699,1486.45 892.699,1486.45 915.805,1486.45 892.699,1486.45 "/>
<path clip-path="url(#clip342)" d="M915.805 1482.72 L915.805 1486.45 L938.91 1486.45 L938.91 1482.72 L915.805 1482.72 L915.805 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="915.805,1482.72 915.805,1486.45 938.91,1486.45 938.91,1482.72 915.805,1482.72 "/>
<path clip-path="url(#clip342)" d="M938.91 1478.99 L938.91 1486.45 L962.015 1486.45 L962.015 1478.99 L938.91 1478.99 L938.91 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="938.91,1478.99 938.91,1486.45 962.015,1486.45 962.015,1478.99 938.91,1478.99 "/>
<path clip-path="url(#clip342)" d="M962.015 1482.72 L962.015 1486.45 L985.121 1486.45 L985.121 1482.72 L962.015 1482.72 L962.015 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="962.015,1482.72 962.015,1486.45 985.121,1486.45 985.121,1482.72 962.015,1482.72 "/>
<path clip-path="url(#clip342)" d="M985.121 1486.45 L985.121 1486.45 L1008.23 1486.45 L1008.23 1486.45 L985.121 1486.45 L985.121 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="985.121,1486.45 985.121,1486.45 1008.23,1486.45 985.121,1486.45 "/>
<path clip-path="url(#clip342)" d="M1008.23 1486.45 L1008.23 1486.45 L1031.33 1486.45 L1031.33 1486.45 L1008.23 1486.45 L1008.23 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1008.23,1486.45 1008.23,1486.45 1031.33,1486.45 1008.23,1486.45 "/>
<path clip-path="url(#clip342)" d="M1031.33 1482.72 L1031.33 1486.45 L1054.44 1486.45 L1054.44 1482.72 L1031.33 1482.72 L1031.33 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1031.33,1482.72 1031.33,1486.45 1054.44,1486.45 1054.44,1482.72 1031.33,1482.72 "/>
<path clip-path="url(#clip342)" d="M1054.44 1478.99 L1054.44 1486.45 L1077.54 1486.45 L1077.54 1478.99 L1054.44 1478.99 L1054.44 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1054.44,1478.99 1054.44,1486.45 1077.54,1486.45 1077.54,1478.99 1054.44,1478.99 "/>
<path clip-path="url(#clip342)" d="M1077.54 1478.99 L1077.54 1486.45 L1100.65 1486.45 L1100.65 1478.99 L1077.54 1478.99 L1077.54 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1077.54,1478.99 1077.54,1486.45 1100.65,1486.45 1100.65,1478.99 1077.54,1478.99 "/>
<path clip-path="url(#clip342)" d="M1100.65 1482.72 L1100.65 1486.45 L1123.75 1486.45 L1123.75 1482.72 L1100.65 1482.72 L1100.65 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1100.65,1482.72 1100.65,1486.45 1123.75,1486.45 1123.75,1482.72 1100.65,1482.72 "/>
<path clip-path="url(#clip342)" d="M1123.75 1486.45 L1123.75 1486.45 L1146.86 1486.45 L1146.86 1486.45 L1123.75 1486.45 L1123.75 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1123.75,1486.45 1123.75,1486.45 1146.86,1486.45 1123.75,1486.45 "/>
<path clip-path="url(#clip342)" d="M1146.86 1486.45 L1146.86 1486.45 L1169.96 1486.45 L1169.96 1486.45 L1146.86 1486.45 L1146.86 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1146.86,1486.45 1146.86,1486.45 1169.96,1486.45 1146.86,1486.45 "/>
<path clip-path="url(#clip342)" d="M1169.96 1482.72 L1169.96 1486.45 L1193.07 1486.45 L1193.07 1482.72 L1169.96 1482.72 L1169.96 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1169.96,1482.72 1169.96,1486.45 1193.07,1486.45 1193.07,1482.72 1169.96,1482.72 "/>
<path clip-path="url(#clip342)" d="M1193.07 1482.72 L1193.07 1486.45 L1216.17 1486.45 L1216.17 1482.72 L1193.07 1482.72 L1193.07 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1193.07,1482.72 1193.07,1486.45 1216.17,1486.45 1216.17,1482.72 1193.07,1482.72 "/>
<path clip-path="url(#clip342)" d="M1216.17 1486.45 L1216.17 1486.45 L1239.28 1486.45 L1239.28 1486.45 L1216.17 1486.45 L1216.17 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1216.17,1486.45 1216.17,1486.45 1239.28,1486.45 1216.17,1486.45 "/>
<path clip-path="url(#clip342)" d="M1239.28 1486.45 L1239.28 1486.45 L1262.39 1486.45 L1262.39 1486.45 L1239.28 1486.45 L1239.28 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1239.28,1486.45 1239.28,1486.45 1262.39,1486.45 1239.28,1486.45 "/>
<path clip-path="url(#clip342)" d="M1262.39 1486.45 L1262.39 1486.45 L1285.49 1486.45 L1285.49 1486.45 L1262.39 1486.45 L1262.39 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1262.39,1486.45 1262.39,1486.45 1285.49,1486.45 1262.39,1486.45 "/>
<path clip-path="url(#clip342)" d="M1285.49 1486.45 L1285.49 1486.45 L1308.6 1486.45 L1308.6 1486.45 L1285.49 1486.45 L1285.49 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1285.49,1486.45 1285.49,1486.45 1308.6,1486.45 1285.49,1486.45 "/>
<path clip-path="url(#clip342)" d="M1308.6 1486.45 L1308.6 1486.45 L1331.7 1486.45 L1331.7 1486.45 L1308.6 1486.45 L1308.6 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1308.6,1486.45 1308.6,1486.45 1331.7,1486.45 1308.6,1486.45 "/>
<path clip-path="url(#clip342)" d="M1331.7 1478.99 L1331.7 1486.45 L1354.81 1486.45 L1354.81 1478.99 L1331.7 1478.99 L1331.7 1478.99  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1331.7,1478.99 1331.7,1486.45 1354.81,1486.45 1354.81,1478.99 1331.7,1478.99 "/>
<path clip-path="url(#clip342)" d="M1354.81 1486.45 L1354.81 1486.45 L1377.91 1486.45 L1377.91 1486.45 L1354.81 1486.45 L1354.81 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1354.81,1486.45 1354.81,1486.45 1377.91,1486.45 1354.81,1486.45 "/>
<path clip-path="url(#clip342)" d="M1377.91 1482.72 L1377.91 1486.45 L1401.02 1486.45 L1401.02 1482.72 L1377.91 1482.72 L1377.91 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1377.91,1482.72 1377.91,1486.45 1401.02,1486.45 1401.02,1482.72 1377.91,1482.72 "/>
<path clip-path="url(#clip342)" d="M1401.02 1482.72 L1401.02 1486.45 L1424.12 1486.45 L1424.12 1482.72 L1401.02 1482.72 L1401.02 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1401.02,1482.72 1401.02,1486.45 1424.12,1486.45 1424.12,1482.72 1401.02,1482.72 "/>
<path clip-path="url(#clip342)" d="M1424.12 1482.72 L1424.12 1486.45 L1447.23 1486.45 L1447.23 1482.72 L1424.12 1482.72 L1424.12 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1424.12,1482.72 1424.12,1486.45 1447.23,1486.45 1447.23,1482.72 1424.12,1482.72 "/>
<path clip-path="url(#clip342)" d="M1447.23 1482.72 L1447.23 1486.45 L1470.33 1486.45 L1470.33 1482.72 L1447.23 1482.72 L1447.23 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1447.23,1482.72 1447.23,1486.45 1470.33,1486.45 1470.33,1482.72 1447.23,1482.72 "/>
<path clip-path="url(#clip342)" d="M1470.33 1486.45 L1470.33 1486.45 L1493.44 1486.45 L1493.44 1486.45 L1470.33 1486.45 L1470.33 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1470.33,1486.45 1470.33,1486.45 1493.44,1486.45 1470.33,1486.45 "/>
<path clip-path="url(#clip342)" d="M1493.44 1486.45 L1493.44 1486.45 L1516.54 1486.45 L1516.54 1486.45 L1493.44 1486.45 L1493.44 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1493.44,1486.45 1493.44,1486.45 1516.54,1486.45 1493.44,1486.45 "/>
<path clip-path="url(#clip342)" d="M1516.54 1482.72 L1516.54 1486.45 L1539.65 1486.45 L1539.65 1482.72 L1516.54 1482.72 L1516.54 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1516.54,1482.72 1516.54,1486.45 1539.65,1486.45 1539.65,1482.72 1516.54,1482.72 "/>
<path clip-path="url(#clip342)" d="M1539.65 1486.45 L1539.65 1486.45 L1562.76 1486.45 L1562.76 1486.45 L1539.65 1486.45 L1539.65 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1539.65,1486.45 1539.65,1486.45 1562.76,1486.45 1539.65,1486.45 "/>
<path clip-path="url(#clip342)" d="M1562.76 1486.45 L1562.76 1486.45 L1585.86 1486.45 L1585.86 1486.45 L1562.76 1486.45 L1562.76 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1562.76,1486.45 1562.76,1486.45 1585.86,1486.45 1562.76,1486.45 "/>
<path clip-path="url(#clip342)" d="M1585.86 1486.45 L1585.86 1486.45 L1608.97 1486.45 L1608.97 1486.45 L1585.86 1486.45 L1585.86 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1585.86,1486.45 1585.86,1486.45 1608.97,1486.45 1585.86,1486.45 "/>
<path clip-path="url(#clip342)" d="M1608.97 1486.45 L1608.97 1486.45 L1632.07 1486.45 L1632.07 1486.45 L1608.97 1486.45 L1608.97 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1608.97,1486.45 1608.97,1486.45 1632.07,1486.45 1608.97,1486.45 "/>
<path clip-path="url(#clip342)" d="M1632.07 1486.45 L1632.07 1486.45 L1655.18 1486.45 L1655.18 1486.45 L1632.07 1486.45 L1632.07 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1632.07,1486.45 1632.07,1486.45 1655.18,1486.45 1632.07,1486.45 "/>
<path clip-path="url(#clip342)" d="M1655.18 1486.45 L1655.18 1486.45 L1678.28 1486.45 L1678.28 1486.45 L1655.18 1486.45 L1655.18 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1655.18,1486.45 1655.18,1486.45 1678.28,1486.45 1655.18,1486.45 "/>
<path clip-path="url(#clip342)" d="M1678.28 1486.45 L1678.28 1486.45 L1701.39 1486.45 L1701.39 1486.45 L1678.28 1486.45 L1678.28 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1678.28,1486.45 1678.28,1486.45 1701.39,1486.45 1678.28,1486.45 "/>
<path clip-path="url(#clip342)" d="M1701.39 1486.45 L1701.39 1486.45 L1724.49 1486.45 L1724.49 1486.45 L1701.39 1486.45 L1701.39 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1701.39,1486.45 1701.39,1486.45 1724.49,1486.45 1701.39,1486.45 "/>
<path clip-path="url(#clip342)" d="M1724.49 1486.45 L1724.49 1486.45 L1747.6 1486.45 L1747.6 1486.45 L1724.49 1486.45 L1724.49 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1724.49,1486.45 1724.49,1486.45 1747.6,1486.45 1724.49,1486.45 "/>
<path clip-path="url(#clip342)" d="M1747.6 1486.45 L1747.6 1486.45 L1770.7 1486.45 L1770.7 1486.45 L1747.6 1486.45 L1747.6 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1747.6,1486.45 1747.6,1486.45 1770.7,1486.45 1747.6,1486.45 "/>
<path clip-path="url(#clip342)" d="M1770.7 1486.45 L1770.7 1486.45 L1793.81 1486.45 L1793.81 1486.45 L1770.7 1486.45 L1770.7 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1770.7,1486.45 1770.7,1486.45 1793.81,1486.45 1770.7,1486.45 "/>
<path clip-path="url(#clip342)" d="M1793.81 1486.45 L1793.81 1486.45 L1816.91 1486.45 L1816.91 1486.45 L1793.81 1486.45 L1793.81 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1793.81,1486.45 1793.81,1486.45 1816.91,1486.45 1793.81,1486.45 "/>
<path clip-path="url(#clip342)" d="M1816.91 1486.45 L1816.91 1486.45 L1840.02 1486.45 L1840.02 1486.45 L1816.91 1486.45 L1816.91 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1816.91,1486.45 1816.91,1486.45 1840.02,1486.45 1816.91,1486.45 "/>
<path clip-path="url(#clip342)" d="M1840.02 1486.45 L1840.02 1486.45 L1863.13 1486.45 L1863.13 1486.45 L1840.02 1486.45 L1840.02 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1840.02,1486.45 1840.02,1486.45 1863.13,1486.45 1840.02,1486.45 "/>
<path clip-path="url(#clip342)" d="M1863.13 1486.45 L1863.13 1486.45 L1886.23 1486.45 L1886.23 1486.45 L1863.13 1486.45 L1863.13 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1863.13,1486.45 1863.13,1486.45 1886.23,1486.45 1863.13,1486.45 "/>
<path clip-path="url(#clip342)" d="M1886.23 1486.45 L1886.23 1486.45 L1909.34 1486.45 L1909.34 1486.45 L1886.23 1486.45 L1886.23 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1886.23,1486.45 1886.23,1486.45 1909.34,1486.45 1886.23,1486.45 "/>
<path clip-path="url(#clip342)" d="M1909.34 1486.45 L1909.34 1486.45 L1932.44 1486.45 L1932.44 1486.45 L1909.34 1486.45 L1909.34 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1909.34,1486.45 1909.34,1486.45 1932.44,1486.45 1909.34,1486.45 "/>
<path clip-path="url(#clip342)" d="M1932.44 1486.45 L1932.44 1486.45 L1955.55 1486.45 L1955.55 1486.45 L1932.44 1486.45 L1932.44 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1932.44,1486.45 1932.44,1486.45 1955.55,1486.45 1932.44,1486.45 "/>
<path clip-path="url(#clip342)" d="M1955.55 1486.45 L1955.55 1486.45 L1978.65 1486.45 L1978.65 1486.45 L1955.55 1486.45 L1955.55 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1955.55,1486.45 1955.55,1486.45 1978.65,1486.45 1955.55,1486.45 "/>
<path clip-path="url(#clip342)" d="M1978.65 1486.45 L1978.65 1486.45 L2001.76 1486.45 L2001.76 1486.45 L1978.65 1486.45 L1978.65 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="1978.65,1486.45 1978.65,1486.45 2001.76,1486.45 1978.65,1486.45 "/>
<path clip-path="url(#clip342)" d="M2001.76 1486.45 L2001.76 1486.45 L2024.86 1486.45 L2024.86 1486.45 L2001.76 1486.45 L2001.76 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2001.76,1486.45 2001.76,1486.45 2024.86,1486.45 2001.76,1486.45 "/>
<path clip-path="url(#clip342)" d="M2024.86 1486.45 L2024.86 1486.45 L2047.97 1486.45 L2047.97 1486.45 L2024.86 1486.45 L2024.86 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2024.86,1486.45 2024.86,1486.45 2047.97,1486.45 2024.86,1486.45 "/>
<path clip-path="url(#clip342)" d="M2047.97 1486.45 L2047.97 1486.45 L2071.07 1486.45 L2071.07 1486.45 L2047.97 1486.45 L2047.97 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2047.97,1486.45 2047.97,1486.45 2071.07,1486.45 2047.97,1486.45 "/>
<path clip-path="url(#clip342)" d="M2071.07 1486.45 L2071.07 1486.45 L2094.18 1486.45 L2094.18 1486.45 L2071.07 1486.45 L2071.07 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2071.07,1486.45 2071.07,1486.45 2094.18,1486.45 2071.07,1486.45 "/>
<path clip-path="url(#clip342)" d="M2094.18 1486.45 L2094.18 1486.45 L2117.28 1486.45 L2117.28 1486.45 L2094.18 1486.45 L2094.18 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2094.18,1486.45 2094.18,1486.45 2117.28,1486.45 2094.18,1486.45 "/>
<path clip-path="url(#clip342)" d="M2117.28 1486.45 L2117.28 1486.45 L2140.39 1486.45 L2140.39 1486.45 L2117.28 1486.45 L2117.28 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2117.28,1486.45 2117.28,1486.45 2140.39,1486.45 2117.28,1486.45 "/>
<path clip-path="url(#clip342)" d="M2140.39 1486.45 L2140.39 1486.45 L2163.5 1486.45 L2163.5 1486.45 L2140.39 1486.45 L2140.39 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2140.39,1486.45 2140.39,1486.45 2163.5,1486.45 2140.39,1486.45 "/>
<path clip-path="url(#clip342)" d="M2163.5 1486.45 L2163.5 1486.45 L2186.6 1486.45 L2186.6 1486.45 L2163.5 1486.45 L2163.5 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2163.5,1486.45 2163.5,1486.45 2186.6,1486.45 2163.5,1486.45 "/>
<path clip-path="url(#clip342)" d="M2186.6 1486.45 L2186.6 1486.45 L2209.71 1486.45 L2209.71 1486.45 L2186.6 1486.45 L2186.6 1486.45  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2186.6,1486.45 2186.6,1486.45 2209.71,1486.45 2186.6,1486.45 "/>
<path clip-path="url(#clip342)" d="M2209.71 1482.72 L2209.71 1486.45 L2232.81 1486.45 L2232.81 1482.72 L2209.71 1482.72 L2209.71 1482.72  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip342)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:0.4; fill:none" points="2209.71,1482.72 2209.71,1486.45 2232.81,1486.45 2232.81,1482.72 2209.71,1482.72 "/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="303.512" cy="65.8866" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="326.618" cy="1218" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="349.723" cy="1355.95" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="372.828" cy="1404.42" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="395.934" cy="1393.24" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="419.039" cy="1449.16" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="442.144" cy="1449.16" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="465.25" cy="1464.08" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="488.355" cy="1441.71" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="511.461" cy="1460.35" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="534.566" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="557.671" cy="1471.53" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="580.777" cy="1467.81" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="603.882" cy="1464.08" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="626.987" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="650.093" cy="1475.26" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="673.198" cy="1475.26" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="696.304" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="719.409" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="742.514" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="765.62" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="788.725" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="811.83" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="834.936" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="858.041" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="881.147" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="904.252" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="927.357" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="950.463" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="973.568" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="996.674" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1019.78" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1042.88" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1065.99" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1089.1" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1112.2" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1135.31" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1158.41" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1181.52" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1204.62" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1227.73" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1250.83" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1273.94" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1297.04" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1320.15" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1343.25" cy="1478.99" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1366.36" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1389.46" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1412.57" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1435.68" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1458.78" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1481.89" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1504.99" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1528.1" cy="1482.72" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1551.2" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1574.31" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1597.41" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1620.52" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1643.62" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1666.73" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1689.83" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1712.94" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1736.05" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1759.15" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1782.26" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1805.36" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1828.47" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1851.57" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1874.68" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1897.78" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1920.89" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1943.99" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1967.1" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="1990.2" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2013.31" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2036.42" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2059.52" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2082.63" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2105.73" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2128.84" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2151.94" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2175.05" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2198.15" cy="1486.45" r="2"/>
<circle clip-path="url(#clip342)" style="fill:#e26f46; stroke:none; fill-opacity:0" cx="2221.26" cy="1482.72" r="2"/>
<path clip-path="url(#clip340)" d="M2009.56 250.738 L2280.06 250.738 L2280.06 95.2176 L2009.56 95.2176  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2009.56,250.738 2280.06,250.738 2280.06,95.2176 2009.56,95.2176 2009.56,250.738 "/>
<path clip-path="url(#clip340)" d="M2033.79 167.794 L2179.17 167.794 L2179.17 126.322 L2033.79 126.322 L2033.79 167.794  Z" fill="#009af9" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2033.79,167.794 2179.17,167.794 2179.17,126.322 2033.79,126.322 2033.79,167.794 "/>
<path clip-path="url(#clip340)" d="M2217.25 166.745 Q2215.44 171.375 2213.73 172.787 Q2212.01 174.199 2209.14 174.199 L2205.74 174.199 L2205.74 170.634 L2208.24 170.634 Q2210 170.634 2210.97 169.8 Q2211.95 168.967 2213.13 165.865 L2213.89 163.921 L2203.4 138.412 L2207.92 138.412 L2216.02 158.689 L2224.12 138.412 L2228.64 138.412 L2217.25 166.745 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2235.93 160.402 L2243.57 160.402 L2243.57 134.037 L2235.26 135.703 L2235.26 131.444 L2243.52 129.778 L2248.2 129.778 L2248.2 160.402 L2255.83 160.402 L2255.83 164.338 L2235.93 164.338 L2235.93 160.402 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2033.79 219.634 L2179.17 219.634 L2179.17 178.162 L2033.79 178.162 L2033.79 219.634  Z" fill="#e26f46" fill-rule="evenodd" fill-opacity="0.4"/>
<polyline clip-path="url(#clip340)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2033.79,219.634 2179.17,219.634 2179.17,178.162 2033.79,178.162 2033.79,219.634 "/>
<path clip-path="url(#clip340)" d="M2217.25 218.585 Q2215.44 223.215 2213.73 224.627 Q2212.01 226.039 2209.14 226.039 L2205.74 226.039 L2205.74 222.474 L2208.24 222.474 Q2210 222.474 2210.97 221.64 Q2211.95 220.807 2213.13 217.705 L2213.89 215.761 L2203.4 190.252 L2207.92 190.252 L2216.02 210.529 L2224.12 190.252 L2228.64 190.252 L2217.25 218.585 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip340)" d="M2239.14 212.242 L2255.46 212.242 L2255.46 216.178 L2233.52 216.178 L2233.52 212.242 Q2236.18 209.488 2240.76 204.858 Q2245.37 200.205 2246.55 198.863 Q2248.8 196.34 2249.68 194.604 Q2250.58 192.844 2250.58 191.155 Q2250.58 188.4 2248.64 186.664 Q2246.71 184.928 2243.61 184.928 Q2241.41 184.928 2238.96 185.692 Q2236.53 186.455 2233.75 188.006 L2233.75 183.284 Q2236.57 182.15 2239.03 181.571 Q2241.48 180.993 2243.52 180.993 Q2248.89 180.993 2252.08 183.678 Q2255.28 186.363 2255.28 190.854 Q2255.28 192.983 2254.47 194.905 Q2253.68 196.803 2251.57 199.395 Q2251 200.067 2247.89 203.284 Q2244.79 206.479 2239.14 212.242 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

```

#-

````@julia
savefig(joinpath(dir, "errors_distribution.png"))
````

````
"/Users/anthony/GoogleDrive/Julia/TumorGrowth/docs/src/examples/04_model_battle/errors_distribution.png"
````

We deem a student t-test inappopriate and instead compute bootstrap confidence intervals
for pairwise differences in model errors:

````@julia
confidence_intervals = Array{Any}(undef, length(models) - 2, length(models) - 2)
for i in 1:(length(models) - 2)
    for j in 1:(length(models) - 2)
        b = bootstrap(
            mean,
            errs[:,i] - errs[:,j],
            BasicSampling(10000),
        )
        confidence_intervals[i,j] = only(confint(b, BasicConfInt(0.95)))[2:3]
    end
end
confidence_intervals
````

````
6×6 Matrix{Any}:
 (0.0, 0.0)                    (2.56833e-5, 0.000859913)    (-4.31786e-5, 0.000838573)   (4.00769e-5, 0.000853559)    (0.000152248, 0.000921184)  (-0.000465458, 0.000405489)
 (-0.000857682, -3.6013e-5)    (0.0, 0.0)                   (-0.000120715, 2.75543e-5)   (-2.59767e-5, 3.76029e-5)    (1.97563e-5, 0.000177436)   (-0.000967472, 4.09321e-6)
 (-0.000839244, 4.63642e-5)    (-3.01687e-5, 0.000123503)   (0.0, 0.0)                   (-5.48039e-5, 0.000153696)   (1.75595e-5, 0.000256715)   (-0.000965075, 6.73056e-5)
 (-0.00084966, -4.01049e-5)    (-3.78307e-5, 2.56403e-5)    (-0.000152721, 5.20309e-5)   (0.0, 0.0)                   (1.64366e-5, 0.000169683)   (-0.000988484, -7.33092e-6)
 (-0.000936245, -0.000162257)  (-0.000179215, -1.78991e-5)  (-0.000254896, -1.99733e-5)  (-0.000170176, -1.25176e-5)  (0.0, 0.0)                  (-0.00104126, -0.000136899)
 (-0.000391998, 0.000464662)   (-6.38049e-6, 0.000960798)   (-7.79031e-5, 0.000948532)   (6.22836e-6, 0.000963191)    (0.000107021, 0.00104599)   (0.0, 0.0)
````

We can interpret the confidence intervals as  follows:

- if both endpoints -ve, row index wins

- if both endpoints +ve, column index wins

- otherwise a draw

````@julia
winner_pointer(ci) = ci == (0, 0) ? "n/a" :
    isnan(first(ci)) && isnan(last(ci)) ? "inconclusive" :
    first(ci) < 0 && last(ci) < 0 ? "←" :
    first(ci) > 0 && last(ci) > 0 ? "↑" :
    "draw"

tabular(A, model_exs) = NamedTuple{(:model, model_exs[2:end]...)}((
    model_exs[1:end-1],
    (A[1:end-1, j] for j in 2:length(model_exs))...,
))

pretty_table(
    tabular(winner_pointer.(confidence_intervals), model_exs[1:5]),
    show_subheader=false,
)
````

````
┌───────────────────────┬──────────┬──────────┬───────────────────────┬─────────────┐
│                 model │ gompertz │ logistic │ classical_bertalanffy │ bertalanffy │
├───────────────────────┼──────────┼──────────┼───────────────────────┼─────────────┤
│           exponential │        ↑ │     draw │                     ↑ │           ↑ │
│              gompertz │      n/a │     draw │                  draw │           ↑ │
│              logistic │     draw │      n/a │                  draw │           ↑ │
│ classical_bertalanffy │     draw │     draw │                   n/a │           ↑ │
└───────────────────────┴──────────┴──────────┴───────────────────────┴─────────────┘

````

## Bootstrap comparison of errors (neural ODE's included)

````@julia
bad_error_rows = filter(axes(errs, 1)) do i
    es = errs[i,:]
    any(isnan, es) || any(isinf, es) || max(es...) > 0.1
end
proportion_bad = length(bad_error_rows)/size(errs, 1)
@show proportion_bad
````

````
0.020440251572327043
````

We remove the additional 2%:

````@julia
good_error_rows = setdiff(axes(errs, 1), bad_error_rows);
errs = errs[good_error_rows,:];
````

And proceed as before, but with all columns of `errs` (all models):

````@julia
confidence_intervals = Array{Any}(undef, length(models), length(models))
for i in 1:length(models)
    for j in 1:length(models)
        b = bootstrap(
            mean,
            errs[:,i] - errs[:,j],
            BasicSampling(10000),
        )
        confidence_intervals[i, j] = only(confint(b, BasicConfInt(0.95)))[2:3]
    end
end

pretty_table(
    tabular(winner_pointer.(confidence_intervals), model_exs),
    show_subheader=false,
    tf=PrettyTables.tf_markdown, vlines=:all,
)
````

````
|                 model | gompertz | logistic | classical_bertalanffy | bertalanffy | bertalanffy2 |   n1 | n2 |
|-----------------------|----------|----------|-----------------------|-------------|--------------|------|----|
|           exponential |        ↑ |     draw |                     ↑ |           ↑ |         draw | draw |  ← |
|              gompertz |      n/a |     draw |                  draw |           ↑ |         draw | draw |  ← |
|              logistic |     draw |      n/a |                  draw |           ↑ |         draw | draw |  ← |
| classical_bertalanffy |     draw |     draw |                   n/a |           ↑ |         draw | draw |  ← |
|           bertalanffy |        ← |        ← |                     ← |         n/a |            ← | draw |  ← |
|          bertalanffy2 |     draw |     draw |                  draw |           ↑ |          n/a | draw |  ← |
|                    n1 |     draw |     draw |                  draw |        draw |         draw |  n/a |  ← |

````

The lack of statistical significance notwithstanding, here are the models, listed in
order of decreasing performance:

````@julia
zipped = collect(zip(models, vec(mean(errs, dims=1))))
sort!(zipped, by=last)
model, error = collect.(zip(zipped...))
rankings = (; model, error)
pretty_table(
    rankings,
    show_subheader=false,
    tf=PrettyTables.tf_markdown, vlines=:all,
)
````

````
|                 model |      error |
|-----------------------|------------|
|           bertalanffy | 0.00272664 |
| classical_bertalanffy | 0.00279946 |
|              gompertz |  0.0028149 |
|              logistic | 0.00288491 |
|    neural (12 params) |  0.0031024 |
|          bertalanffy2 | 0.00318344 |
|           exponential | 0.00331202 |
|   neural2 (14 params) |  0.0045919 |

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

