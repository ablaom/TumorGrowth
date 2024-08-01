using Test
using TumorGrowth
import StableRNGs.StableRNG
using Tables
using ComponentArrays

const component = ComponentArray

@testset "slope" begin
    rng = StableRNG(123)
    xs = rand(rng, 3)
    m, c  = rand(rng, 2)
    ys = m*xs .+ c
    @test TumorGrowth.slope(xs, ys) ≈ m
end

@testset "curvature" begin
    rng = StableRNG(123)
    xs = rand(rng, 3)
    a, b, c  = rand(rng, 3)
    ys = a*xs.^2 + b*xs .+ c
    @test TumorGrowth.curvature(xs, ys) ≈ a
end

@testset "WeightedL2Loss" begin
    ŷ = [2.3, 7.6]
    times = [5.6, 8.9]
    loss = TumorGrowth.WeightedL2Loss()
    weighted_loss = TumorGrowth.WeightedL2Loss(times, 8.9 - 5.6)
    @test loss(ŷ, zeros(2))          ≈     2.3^2 + 7.6^2
    @test weighted_loss(ŷ, zeros(2)) ≈ 0.5*2.3^2 + 7.6^2
end

@testset "recover" begin
    tup = (x=1, y=nothing, z=3, w=nothing)
    from = (x=10, y=2)
    @test TumorGrowth.recover(tup, from) == (x=1, y=2, z=3, w=nothing)
end

@testset "constraints" begin
    lower = NamedTuple{(:x, :z)}((1, 3))
    upper = NamedTuple{(:x, :z)}((10, 30))
    x_candidate = NamedTuple{(:x, :y, :z)}((0, 11, 31)) |> component
    x = NamedTuple{(:x, :y, :z)}((3, 12, 26))
    @test !TumorGrowth.satisfies_constraints(x_candidate, lower, upper)
    @test collect(TumorGrowth.force_constraints!(x_candidate, x, lower, upper)) ≈
        [2.0, 11.0, 28.0]
    @test TumorGrowth.satisfies_constraints(x_candidate, lower, upper)

    namedtuple(v) = Tables.rowtable(Tables.table(v')) |> first
    # julia> namedtuple([10, 20, 30])
    # (Column1 = 10, Column2 = 20, Column3 = 30)

    rng = StableRNG(123)
    lower = rand(rng, 50) |> namedtuple |> component
    upper = lower + rand(rng, 50) |> namedtuple |> component
    x = lower + rand(rng, 50).*(upper - lower) |> namedtuple
    x_candidate = rand(rng, 50) |> namedtuple |> component
    @test !TumorGrowth.satisfies_constraints(x_candidate, lower, upper)
    TumorGrowth.force_constraints!(x_candidate, x, lower, upper);
    @test TumorGrowth.satisfies_constraints(x_candidate, lower, upper)
end

@testset "delete" begin
    @test TumorGrowth.delete(component((x=1, y=2, z=3)), (:y,)) == (x=1, z=3)
end


@testset "fill_gaps" begin
    long = (a=1, b=rand(1,2), c=(d=4, e=rand(2))) |> component
    short = (; a=10)
    @test TumorGrowth.fill_gaps(short, long, Inf) ==
        (a=10, b=[Inf Inf], c=(d=Inf, e=[Inf, Inf])) |> component
end

true
