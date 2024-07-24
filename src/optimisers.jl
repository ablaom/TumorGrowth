# # OPTIMISATION PROBLEM FOR GENERAL OBJECTIVE FUNCTION

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
            @warn "Solution out of bounds; solution reverted "*
                "to last in-bounds value. Perhaps try a "*
                "smaller `learning_rate`, larger `penalty`, "*
                "or freeze some components. Out of bounds value:"*
                "\n$x"
            break
        end
    end
    problem.x
end

IterationControl.loss(problem::OptimisationProblem) =
    problem.inbounds ? problem.f(problem.x) : Inf
_unwrap(sol) = sol
_unwrap(sol::ComponentArray) = TumorGrowth.functor(sol) |> first
solution(problem::OptimisationProblem) = _unwrap(problem.x)


# # GAUSS-NEWTON VARIATIONS OF LEAST SQUARES OPTIMIZATION

# The following is a wrapper for `LSO.LeastSquaresProblem`, where `LSO=LeastSquaresOptim`.

# In the wrapper, `problem::LSO.LeastSquaresProblem` is assumed to be formulated for an
# objective function of `xfree`, which excludes frozen parameters. We need `reconstruct`
# to recover a full parameter `x` with the frozen entries reinstated (as a
# `ComponentArray`).

mutable struct GaussNewtonProblem{G,P,O}
    g::G # `g(x) == false` means `x` out of bounds
    reconstruct # `reconstruct(xfree)` restores the frozen entries
    problem::P
    optimiser::O  # has type `LSO.Dogleg` or `LSO.LevenbergMarquardt`
    loss::Float64
    function GaussNewtonProblem(
        g::G,
        reconstruct,
        problem::P,
        optimiser::O,
        ) where {G,P,O}
        new{G,P,O}(g, reconstruct, problem, optimiser, Inf)
    end
end

# expose solution as a property, `x`:
function Base.getproperty(p::GaussNewtonProblem, name::Symbol)
    if name == :x
        getfield(p, :problem).x
    else
        getfield(p, name)
    end
end

function Base.show(io::IO, problem::GaussNewtonProblem)
    print(io, "GaussNewtonProblem: \n  "*
        "optimiser: $(problem.optimiser)\n  "*
        "type of solutions: $(typeof(problem.x))"
          )
    return
end

function IterationControl.train!(gnproblem::GaussNewtonProblem, n)
    # `loss` is ignored here:
    @unpack g, reconstruct, problem, optimiser, loss = gnproblem
    if n == 0
        x_previous = problem.x
        solution = LSO.optimize!(problem, optimiser)
        if !g(reconstruct(problem.x))
            gnproblem.inbounds=false
            @warn "Solution out of bounds; solution reverted "*
                "to initial value. Try specifying a specific number of iterations. "
            problem.x = x_previous
            gnproblem.loss = Inf
        else
            gnproblem.loss = solution.ssr
        end
        return problem.x
    end

    for _ in 1:n
        x_previous = problem.x
        solution = LSO.optimize!(problem, optimiser, iterations=1)
        if !g(reconstruct(problem.x))
            gnproblem.inbounds=false
            @warn "Solution out of bounds; solution reverted "*
                "to last inbounds value. "
            problem.x = x_previous
            gnproblem.loss = Inf
            break
        end
        gnproblem.loss = solution.ssr
    end
    return problem.x
end
IterationControl.train!(gnproblem::GaussNewtonProblem) =
    IterationControl.train!(gnproblem, 0)
IterationControl.loss(gnproblem::GaussNewtonProblem) = gnproblem.loss

solution(gnproblem::GaussNewtonProblem) = _unwrap(gnproblem.reconstruct(gnproblem.x))


# # SPECIAL CASE OF A PARAMETERIZED FUNCTION OF ONE VARIABLE

# We specialize to the case of minimizing the least squares error for a parameterized,
# real-valued function of one variable, `x -> F(x, p)` (`p` a vector of parameters), given
# some ground truth data. That is, given points in the plane `(xᵢ, yᵢ)`, we seek
# parameters `p` that minimizes `Σᵢ(yᵢ - F(xᵢ, p))²` for some prescribed `xᵢ` and `yᵢ`.
# Instead, a more general `loss(x, y, p)` function can be provided and we minimize
# `loss(x, F(x, p; kwargs...), p)`, where `kwargs` are user-specified.

# Warning: In `OptimisationProblem` code above, `x` is the solution to optimisation
# problem, but that is here denoted by `p` because `x` refers to ordinate in the plane.

# Optimisation is using gradient descent (`problem` will be an instance of
# `OptimisationProblem` above) or using a specialized Levenber-Marquardt/Dogleg,
# least-squares optimiser (`problem`) will be an instance of
# `LevenbergMaquardt`).

mutable struct CurveOptimisationProblem{T<:Number,FF,P}
    xs::AbstractVector{T}
    ys::AbstractVector{T}
    F::FF
    problem::P
    function CurveOptimisationProblem(
        xs::AbstractVector{T},     # abscissae
        ys::AbstractVector{T},     # corresponding ground truth ordinates
        F::FF,  # `F(xs, p)` are curve ordinates for abscissae `xs` and parameter `p`
        p;  # initial guess for `p`; will be a `ComponentArray`
        loss = TumorGrowth.WeightedL2Loss(), # ignored in GaussNewton case
        learning_rate=0.0,                   # ignored in GaussNewton case
        # Next argument Optimisers.jl optimiser or has type LSO.Dogleg or
        # LSO.LevenbergMarquardt:
        optimiser=Optimisers.Adam(learning_rate),
        g = _->true,
        frozen=NamedTuple(),
        scale=identity, # ignored in GaussNewton case
        kwargs...,
        ) where {T,FF}

        n = length(xs)
        n == length(ys) || throw(ArgumentError(
            "Abscissae `xs` and ordinates `ys` must be equal in number. "
        ))

        if optimiser isa Union{LSO.LevenbergMarquardt,LSO.Dogleg}
            learning_rate == 0 ||
                @warn "Optimiser is `$optimiser`, so ignoring `learning_rate`. "
            loss isa TumorGrowth.WeightedL2Loss{Nothing} ||
                @warn "Optimiser is `$optimiser`, so all is data weighted equally. "

            # `pfree` has the keys that appear in `frozen` removed; `reconstruct` lets us
            # restore the frozen entries:
            pfree, reconstruct = TumorGrowth.functor(p, frozen)
            cfree = ComponentArray(pfree)
            h(cfree) = F(xs, reconstruct(cfree))

            # in-place version of function whose output components get minimized, in the
            # sense of a sum of squares:
            f!(out, cfree) = copy!(out, h(cfree) .- ys)

            # in-place version of Jacobian of that function:
            g!(J, cfree) = copy!(
                J,
                Zygote.jacobian(h, cfree) |> first,
            )

            least_squares_problem =
                LSO.LeastSquaresProblem(; x = cfree, f!, g!, output_length = length(xs))

            problem = GaussNewtonProblem(
                g,
                reconstruct,
                least_squares_problem,
                optimiser,
            )
        else
            f(p) = loss(F(xs, p; kwargs...), ys, p)
            problem = OptimisationProblem(f, p; learning_rate, optimiser, g, frozen, scale)
        end

        return new{T,FF,typeof(problem)}(xs, ys, F, problem)
    end
end

function Base.show(io::IO, problem::CurveOptimisationProblem{T,FF,P}) where {T,FF,P}
    if P <: GaussNewtonProblem{<:Any,<:Any,<:LSO.Dogleg}
        algorithm = "dogleg"
    elseif P <: GaussNewtonProblem{<:Any,<:Any,<:LSO.LevenbergMarquardt}
        algorithm = "Levenberg/Marquardt"
    else
        algorithm = "gradient descent"
    end
    print(
        io,
        "CurveOptimisationProblem: \n  "*
        "algorithm: $algorithm",
        )
end

IterationControl.loss(c::CurveOptimisationProblem) =
    IterationControl.loss(c.problem)
function IterationControl.train!(c::CurveOptimisationProblem, n)
    problem = c.problem
    solution = IterationControl.train!(problem, n)
    c.problem = problem
    return solution
end

solution(c::CurveOptimisationProblem) =
    solution(c.problem)

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
