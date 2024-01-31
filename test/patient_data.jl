using Test
using TumorGrowth

const features = Symbol.([
    "Pt_hashID",
    "Study_Arm",
    "Study_id",
    "Arm_id",
    "T_weeks",
    "T_days",
    "Lesion_diam",
    "Lesion_vol",
    "Lesion_normvol",
    "response",
    "readings",
])

const first_row = [
    "0218075314855e6ceacca856fcd4c737-S1",
    "Study_1_Arm_1",
    1,
    1,
    0.1,
    -21,
    13,
    1142.44,
    0.000185364052636979,
    "flux",
    7,
]

flat_table = flat_patient_data()

@test keys(flat_table[1]) == Tuple(features)
@test values(flat_table[1]) == Tuple(first_row)

table = patient_data();
record = table[1]
@test record.Pt_hashID == first_row[1]
@test keys(record) == Tuple(features)
@test size(record.Lesion_vol) == (7, )
@test record.Lesion_diam isa Vector{<:Real}

true
