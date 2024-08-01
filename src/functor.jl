# recursive deconstructor for ComponentArray:

"""
    TumorGrowth.functor(x) -> destructured_x, recover

**Private method.**

An extension of `Functors.functor` from the package Functors.jl, with an
overloading for `ComponentArray`s.

"""
functor(x) = Functors.functor(x)
_functor(c) = c
_functor(c::ComponentArray) =
    NamedTuple{propertynames(c)}(_functor.(getproperty.((c,), propertynames(c))))
functor(c::ComponentArray) = _functor(c), ComponentArray

"""
    function TumorGrowth.functor(x, frozen)

**Private method.**

For a `ComponentArray`, `x`, return a tuple `(xfree, reconstructor)`, where:

- `xfree` is a deconstructed version of `x` with entries corresponding to keys in the
  ordinary named tuple `frozen` deleted.

- `reconstruct` is a method to reconstruct a `ComponentArray` from something similar to
  `xfree`, ensuring the missing keys get values from the named tuple `frozen`, as
  demonstrated in the example below. You can also apply `reconstruct` to things like
  `xfree` wrapped as `ComponentArray`s.

```julia

c = (x =1, y=2, z=3) |> ComponentArray
free, reconstruct = TumorGrowth.functor(c, (; y=20))
julia> free
(x = 1, z = 3)

julia> reconstruct((x=100, z=300))
ComponentVector{Int64}(x = 100, y = 20, z = 300)

julia> reconstruct(ComponentArray(x=100, z=300)))
ComponentVector{Int64}(x = 100, y = 20, z = 300)
```

"""
function functor(x, frozen)
    xfree = TumorGrowth.delete(x, keys(frozen))
    reconstructor(xfree) = merge(x, merge(xfree, frozen))
    return xfree, reconstructor
end
