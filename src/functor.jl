# recursive deconstructor for ComponentArray:

functor(x) = Functors.functor(x)
_functor(c) = c
_functor(c::ComponentArray) =
    NamedTuple{propertynames(c)}(_functor.(getproperty.((c,), propertynames(c))))
functor(c::ComponentArray) = _functor(c), ComponentArray
