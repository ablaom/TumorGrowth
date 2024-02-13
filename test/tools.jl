using Test
using TumorGrowth
import StableRNGs.StableRNG

# @testset "slope" begin
#     rng = StableRNG(123)
#     xs = rand(rng, 3)
#     m, c  = rand(rng, 2)
#     ys = m*xs .+ c
#     @test TumorGrowth.slope(xs, ys) ≈ m
# end

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

true
