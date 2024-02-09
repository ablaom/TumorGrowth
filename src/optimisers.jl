# # OPTIMISATION PROBLEMS

const ERR_OUT_OF_BOUNDS = ArgumentError( "Initial candidate solution is out of bounds. " )

# An *optimisation problem* consists of both the data specifying some optimisation problem
# (objective function, `f`, etc) and an updatable candidate solution, `x`, solving the
# problem (minimizing `f(x)`).

# To control the optimisation we implement the
# [IterationControl.jl](https://github.com/JuliaAI/IterationControl.jl/tree/dev/src)
# interface for our optimisation problems.

# For the moment we are restricting to gradient descent optimisers, which we get from
# Optimisers.jl.

mutable struct OptimisationProblem{F,X,R,G,O,XF,SC,S}
    f::F # objective function with values of type `T`
    x::X # candidate solution
    reconstruct::R
    g::G # `g(x) == false` means `x` out of bounds
    optimiser::O
    frozen::XF  # frozen part of candidate solution
    scale::SC # `scale(x)` is `x` with each component multiplied by
              # a scaling factor
    inbounds::Bool
    state::S
end
function OptimisationProblem(
    f,
    x_user;
    g = x-> true,
    learning_rate=0.01,
    optimiser=Optimisers.Adam(learning_rate),
    frozen=NamedTuple(),
    scale=identity,
    )

    x = deepcopy(x_user)
    _, reconstruct = TumorGrowth.functor(x)
    g(x) || throw(ERR_OUT_OF_BOUNDS)
    state = Optimisers.setup(optimiser, x)
    return OptimisationProblem(
        f,
        x,
        reconstruct,
        g,
        optimiser,
        frozen,
        scale,
        true,
        state,
    )
end

function Base.show(io::IO, problem::OptimisationProblem)
    scaling = problem.scale == identity ? "no" : "yes"
    print(io, "OptimisationProblem: \n  "*
        "optimiser: $(problem.optimiser)\n  "*
        "type of solutions: $(typeof(problem.x))\n  "*
        "frozen: $(problem.frozen)\n  "*
        "scaling? $scaling"
          )
    return
end

function IterationControl.train!(problem::OptimisationProblem, n)
    @unpack f, x, reconstruct, g, optimiser, frozen, scale, inbounds, state = problem
    frozen_nt, _ = TumorGrowth.functor(frozen)
    for _ in 1:n
        ∇f = Sens.Zygote.gradient(f, x) |> only
        ∇f = scale(scale(∇f))
        state, x = Optimisers.update(state, x, ∇f)
        x_nt, _ = TumorGrowth.functor(x)
        x_candidate = merge(x_nt, frozen_nt) |> reconstruct
        if g(x_candidate)
            problem.x = x_candidate
            problem.state = state
        else
            problem.inbounds=false
            @warn "Returning `Inf` loss, as solution is out of bounds; solution reverted "*
                "to last inbounds value. Perhaps try a "*
                "smaller `learning_rate`, larger `penalty`, "*
                "or freeze some components. Out of bounds value:"*
                "\n$x"
        end
    end
    problem.x
end

IterationControl.loss(problem::OptimisationProblem) =
    problem.inbounds ? problem.f(problem.x) : Inf
_unwrap(sol) = sol
_unwrap(sol::ComponentArray) = TumorGrowth.functor(sol) |> first
solution(problem::OptimisationProblem) = _unwrap(problem.x)


# # SPECIAL CASE OF A PARAMETERIZED FUNCTION OF ONE VARIABLE

# We specialize to the case of minimizing the least squares error for a parameterized,
# real-valued function of one variable, `x -> F(x, p)` (`p` a vector of parameters), given
# some ground truth data. That is, given points in the plane `(xᵢ, yᵢ)`, we seek
# parameters `p` that minimizes `Σᵢ(yᵢ - F(xᵢ, p))²` for some prescribed `xᵢ` and `yᵢ`.
# Instead, a more general `loss(x, y, p)` function can be provided and we minimize
# `loss(x, F(x, p; kwargs...), p)`, where `kwargs` are user-specified.

# Warning: In `OptimisationProblem` code above, `x` is the solution to optimisation
# problem, but that is here denoted by `p` because `x` refers to ordinate in the plane.

mutable struct CurveOptimisationProblem{T<:Number,FF,O}
    xs::AbstractVector{T}
    ys::AbstractVector{T}
    F::FF
    optimisation_problem::O
    function CurveOptimisationProblem(
        xs::AbstractVector{T},     # abscissae
        ys::AbstractVector{T},     # corresponding ground truth ordinates
        F::FF,  # `F(xs, p)` are curve ordinates for abscissae `xs` and parameter `p`
        p;  # initial guess for `p`
        loss = TumorGrowth.WeightedL2Loss(),
        learning_rate=0.0001,
        optimiser=Optimisers.Adam(learning_rate),
        g = _->true,
        frozen=NamedTuple(),
        scale=identity,
        kwargs...,
        ) where {T,FF}

        n = length(xs)
        n == length(ys) || throw(ArgumentError(
            "Abscissae `xs` and ordinates `ys` must be equal in number. "
        ))

        f(p) = loss(F(xs, p; kwargs...), ys, p)

        problem = OptimisationProblem(f, p; learning_rate, optimiser, g, frozen, scale)
        return new{T,FF,typeof(problem)}(xs, ys, F, problem)
    end
end

IterationControl.loss(c::CurveOptimisationProblem) =
    IterationControl.loss(c.optimisation_problem)
function IterationControl.train!(c::CurveOptimisationProblem, n)
    problem = c.optimisation_problem
    solution = IterationControl.train!(problem, n)
    c.optimisation_problem = problem
    return solution
end

solution(c::CurveOptimisationProblem) =
    solution(c.optimisation_problem)

# exported names:

"""
    solve!(problem, n)

Solve a calibration `problem`, as constructed with [`CalibrationProblem`](@ref). The
calibrated parameters are then returned by `solution(problem)`.

---

    solve!(problem, controls...)

Solve a calibration `problem` using one or more iteration `controls`, from the package
IterationControls.jl. See the "Extended help" section of [`CalibrationProblem`](@ref) for
examples.

"""
solve!(args...; kwargs...) = IterationControl.train!(args...; kwargs...)

"""
    loss(problem)

Return the sum of squares loss for a calibration `problem`, as constructed with
[`CalibrationProblem`](@ref).

"""
loss(args...; kwargs...) = IterationControl.loss(args...; kwargs...)
