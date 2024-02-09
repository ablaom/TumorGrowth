"""
    slope(xs, ys)

Return the slope of the line of least-squares best fit for ordinates `xs` and coordinates
`ys`.

"""
function slope(xs::AbstractArray{T}, ys) where T
    A = hcat(xs, ones(T, length(xs)))
    At = adjoint(A)
    m, c = (At*A) \ At*ys
    return m
end

"""
    curvature(xs, ys)

Return the coefficient `a` for the parabola `x -> a*x^2 + b*x + c` of best
fit, for ordinates `xs` and coordinates `ys`.

"""
function curvature(xs::AbstractArray{T}, ys) where T
    A = hcat(xs.^2, xs, ones(T, length(xs)))
    At = adjoint(A)
    a, b, c = (At*A) \ At*ys
    return a
end

"""
    WeightedL2Loss(times, h=Inf)

*Private method.*

Return a weighted sum of squares loss function `(ŷ, y) -> loss`, where the weights decay
in reverse time with a half life `h`.

"""
struct WeightedL2Loss{T}
    weights::Vector{T}
    function WeightedL2Loss(times, h)
        h > 0 || error("Need positive half life. Set to `Inf` for no decay. ")
        h == Inf && return new{Nothing}([nothing,])
        t_end = times[end]
        weights = map(times) do t
            exp(log(2)*(t - t_end)/h)
        end
        return new{eltype(weights)}(weights)
    end
end
WeightedL2Loss(times) = WeightedL2Loss(times, Inf)
WeightedL2Loss() = WeightedL2Loss(Float64[])
(loss::WeightedL2Loss{Nothing})(ŷ, y) = (ŷ .- y).^2 |> sum
(loss::WeightedL2Loss)(ŷ,  y) = loss.weights .* (ŷ .- y).^2 |> sum
(loss::WeightedL2Loss)(ŷ, y, p) = loss(ŷ, y)


grab(x, y) = x
grab(x::Nothing, y) = y

"""
    recover(tuple, from)

*Private method.*

Return a new tuple by replacing any `nothing` values with the corresponding value in the
`from` tuple, whenever a corresponding key exists, and otherwise not make the replacement.

```julia
julia> recover((x=1, y=nothing, z=3, w=nothing), (x=10, y=2, k=7))
(x = 1, y = 2, z = 3, w = nothing)
```

"""
function recover(tup, from)
    names = keys(tup)
    vals = values(tup)
    new_vals = map(names) do name
        grab(getproperty(tup, name), get(from, name, nothing))
    end
    return NamedTuple{names}(new_vals)
end
