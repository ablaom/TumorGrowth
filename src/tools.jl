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
        h == Inf && return new{Nothing}(Nothing[])
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

Return a new named tuple by replacing any `nothing` values with the corresponding value in
the `from` named tuple, whenever a corresponding key exists, and otherwise ignore.

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

"""
    TumorGrowth.merge(x, y::NamedTuple)

*Private method.*

Ordinary merge if `x` is also a named tuple. More generally, first deconstruct `x` using
`TumorGrowth.functor`, merge as usual, and reconstruct.

"""
merge(x, y) = Base.merge(x, y)
function merge(x::ComponentArray, y)
    p, reconstruct = TumorGrowth.functor(x)
    return merge(p, y) |> reconstruct
end

"""
    delete(x, kys)

**Private method.**

Assuming `x` is a named tuple, return a copy of `x` with any key in `kys`
removed. Otherwise, assuming `x` is a structured object (such as a `ComponentArray`) first
convert to a named tuple and then delete the specified keys.

"""
function delete(x::NamedTuple, kys)
    keep = filter(keys(x)) do k
        !(k in kys)
    end
    values = map(keep) do k
        getproperty(x, k)
    end
    NamedTuple{keep}(values)
end

function delete(x, kys)
    p, _ = TumorGrowth.functor(x)
    delete(p, kys)
end

"""
    satisfies_constraints(x, lower, upper)

**Private method.**

Returns `true` if both of the following are true:

- `upper.k < x.k` for each `k` appearing as a key of `upper`
- `x.k < lower.k` for each `k` appearing as a key of `lower`

Otherwise, returns `false`.

"""
function satisfies_constraints(x, lower, upper)
    any(keys(lower)) do k
        getproperty(x, k) ≤ getproperty(lower, k)
    end && return false
    any(keys(upper)) do k
        getproperty(x, k) ≥ getproperty(upper, k)
    end && return false
    return true
end

"""
    force_constraints!(x_candidate, x, lower, upper)

**Private method.**

Assumes `x` is a `ComponentArray` for which [`TumorGrowth.satisfies_constraints(x, lower,
upper)`](@ref) is `true`. The method mutates those components of the `x_candidate` which
do not satisfy the constraints by moving from `x` towards the boundary half the distance
to the boundary, along the failed component.

"""
function force_constraints!(x_candidate, x, lower, upper)
    for k in keys(lower)
        L = getproperty(lower, k)
        if getproperty(x_candidate, k) ≤ L
            setproperty!(x_candidate, k, (L + getproperty(x, k))/2)
        end
    end
    for k in keys(upper)
        U = getproperty(upper, k)
        if getproperty(x_candidate, k) ≥ U
            setproperty!(x_candidate, k, (U + getproperty(x, k))/2)
        end
    end
    return x_candidate
end

instead(::Number, filler) = filler
instead(a::AbstractArray, filler) = instead.(a, Ref(filler))

"""
    fill_gaps(short, long, filler)

**Private method.**

Here `long` is a `ComponentArray` and `short` a named tuple with some of the keys from
`long`. The method returns a `ComponentArray` with the same structure as `long` but with
the values of `short` merged into `long`, with all other (possibly nested) values replaced
with `filler` for numerical values or arrays of `Inf` in the case of array values.

```julia
long = (a=1, b=rand(1,2), c=(d=4, e=rand(2))) |> ComponentArray
short = (; a=10) # could alternatively be a `ComponentArray`

julia> TumorGrowth.fill_gaps(short, long, Inf)
filled = ComponentVector{Float64}(a = 10.0, b = [Inf Inf], c = (d = Inf, e = [Inf, Inf]))

julia> all(filled .> long)
true

"""
function fill_gaps(short, long, filler)
    long_nt, reconstruct = TumorGrowth.functor(Functors.fmap(t->instead(t, filler), long))
    return merge(long_nt, first(TumorGrowth.functor(short))) |> reconstruct
end

const SUCCESS_RETURN_CODES = map([:Default, :Success]) do code
    :(Sens.SciMLBase.ReturnCode.$code) |> eval
end

is_okay(solution) = solution.retcode in SUCCESS_RETURN_CODES
