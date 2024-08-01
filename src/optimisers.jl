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

mutable struct OptimisationProblem{F,X,L,U,R,O,XF,SC,S}
    f::F # objective function with values of type `T`
    x::X # candidate solution
    reconstruct::R
    lower # lower bound constraint on x
    upper # upper bound constraint on x
    optimiser::O
    frozen::XF  # frozen part of candidate solution
    scale::SC # `scale(x)` is `x` with each component multiplied by
              # a scaling factor
    inbounds::Bool
    state::S
    function OptimisationProblem(
        f::F,
        x_user::X,
        lower::L,
        upper::U,
        optimiser::O,
        frozen::XF,
        scale::SC,
        ) where {F,X,L,U,O,XF,SC}

        TumorGrowth.satisfies_constraints(x_user, lower, upper) || throw(ERR_OUT_OF_BOUNDS)
        x = deepcopy(x_user)
        _, reconstruct = TumorGrowth.functor(x)
        state = Optimisers.setup(optimiser, x)
        return new{F,X,L,U,typeof(reconstruct),O,XF,SC,typeof(state)}(
            f,
            x,
            reconstruct,
            lower,
            upper,
            optimiser,
            frozen,
            scale,
            true,
            state,
        )
    end
end

function Base.show(io::IO, problem::OptimisationProblem)
    scaling = problem.scale == identity ? "no" : "yes"
    print(io, "OptimisationProblem: \n  "*
        "optimiser: $(problem.optimiser)\n  "*
        "parameters: $keys(problem.x)\n  "*
        "frozen: $(problem.frozen)\n  "*
        "scaling? $scaling"
          )
    return
end

function IterationControl.train!(problem::OptimisationProblem, n)
    @unpack f, x, reconstruct, lower, upper, optimiser, frozen, scale, inbounds, state = problem
    frozen_named_tuple, _ = TumorGrowth.functor(frozen)
    s(x) = reconstruct(scale(x))
    for _ in 1:n
        x = problem.x
        ∇f = Sens.Zygote.gradient(f, x) |> only
        ∇f = s(s(∇f))
        state, x_candidate = Optimisers.update(state, x, ∇f)
        x_named_tuple, _ = TumorGrowth.functor(x_candidate)
        x_candidate = merge(x_named_tuple, frozen_named_tuple) |> reconstruct
        TumorGrowth.force_constraints!(x_candidate, x, lower, upper)
        problem.x = x_candidate
        problem.state = state
    end
    problem.x
end

IterationControl.loss(problem::OptimisationProblem) =
    problem.inbounds ? problem.f(problem.x) : Inf
_unwrap(sol) = sol
_unwrap(sol::ComponentArray) = TumorGrowth.functor(sol) |> first
solution(problem::OptimisationProblem) = _unwrap(problem.x)


# # GAUSS-NEWTON VARIATIONS OF LEAST SQUARES OPTIMIZATION

# We define a wrapper for `LSO.LeastSquaresProblem`, where `LSO=LeastSquaresOptim`.

# In the wrapper, `problem::LSO.LeastSquaresProblem` is assumed to be formulated for an
# objective function of `xfree`, which excludes frozen parameters. We need `reconstruct`
# to recover a full parameter `x` with the frozen entries reinstated (as a
# `ComponentArray`).


const GaussNewtonOptimiser = Union{LSO.LevenbergMarquardt,LSO.Dogleg}

mutable struct GaussNewtonProblem{
    P <: LSO.LeastSquaresProblem,
    L,
    U,
    O <: GaussNewtonOptimiser,
    R,
    }
    problem::P  # LSO.LeastSquaresProblem whose `x` field excludes frozens
    lower::L
    upper::U
    Δ::Float64
    optimiser::O   # has type `LSO.Dogleg` or `LSO.LevenbergMarquardt`
    reconstruct::R # `reconstruct(xfree)` restores the frozen entries
    loss::Float64
    function GaussNewtonProblem(
        problem::P,
        lower,
        upper,
        Δ,
        optimiser::O,
        reconstruct::R,
        ) where {P,O,R}

        # need to extend bounds to full-blown component arrays, matching the
        # structure of `problem.x`, unless empty:
        l = isempty(lower) ? Float64[] : TumorGrowth.fill_gaps(lower, problem.x, -Inf)
        u = isempty(upper) ? Float64[] : TumorGrwoth.fill_gaps(upper, problem.x, Inf)
        new{P,typeof(l),typeof(u),O,R}(problem, l, u, Δ, optimiser, reconstruct, Inf)
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
        "parameters: $(keys(problem.x))"
          )
    return
end

function IterationControl.train!(gnproblem::GaussNewtonProblem, n)
    # `loss` is ignored here:
    @unpack problem, lower, upper, Δ, optimiser, reconstruct, loss = gnproblem

    options = (; lower, upper, Δ)
    if n > 0
        options = merge(options, (; iterations=n))
    end
    solution = LSO.optimize!(problem, optimiser; options...)
    gnproblem.loss = solution.ssr
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
        lower=(;),
        upper=(;),
        optimiser=Optimisers.Adam(learning_rate),
        Δ = 10.0,
        frozen=NamedTuple(),
        scale=identity, # ignored in GaussNewton case
        kwargs...,
        ) where {T,FF}

        n = length(xs)
        n == length(ys) || throw(ArgumentError(
            "Abscissae `xs` and ordinates `ys` must be equal in number. "
        ))

        if optimiser isa GaussNewtonOptimiser

            # `pfree` has the keys that appear in `frozen` removed; `reconstruct` lets us
            # restore the frozen entries:
            pfree, reconstruct = TumorGrowth.functor(p, frozen)
            cfree = ComponentArray(pfree)
            h(cfree) = F(xs, reconstruct(cfree))

            # we need to remove the frozen keys from `lower` and `upper` as well:
            kys = keys(frozen)
            lower = delete(lower, kys)
            upper = delete(upper, kys)

            # in-place version of function whose output components get minimized, in the
            # sense of a sum of squares:
            f!(out, cfree) = copy!(out, h(cfree) .- ys)

            # in-place version of Jacobian of that function:
            g!(J, cfree) = copy!(
                J,
                Sens.Zygote.jacobian(h, cfree) |> first,
            )

            least_squares_problem =
                LSO.LeastSquaresProblem(; x = cfree, f!, g!, output_length = length(xs))

            problem = GaussNewtonProblem(
                least_squares_problem,
                lower,
                upper,
                Δ,
                optimiser,
                reconstruct,
            )
        else
            f(p) = loss(F(xs, p; kwargs...), ys, p)
            problem = OptimisationProblem(
                f,
                p,
                lower,
                upper,
                optimiser,
                frozen,
                scale,
            )
        end

        return new{T,FF,typeof(problem)}(xs, ys, F, problem)
    end
end

optimiser(problem::CurveOptimisationProblem) = problem.problem.optimiser

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
            "algorithm: $algorithm\n  "*
            "parameters: $(keys(solution(problem)))"
        )
end

const ERR_ZERO_ITERATIONS = ArgumentError(
    "You must specify a positive number of iterations"
)

IterationControl.loss(c::CurveOptimisationProblem) =
    IterationControl.loss(c.problem)
function IterationControl.train!(c::CurveOptimisationProblem, n)
    c.problem isa OptimisationProblem && n == 0 &&
        throw(ERR_ZERO_ITERATIONS)
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

If using a Gauss-Newton optimiser (`LevenbergMarquardt` or `Dogleg`) specify `n=0` to
choose `n` automatically.

---

    solve!(problem, controls...)

Solve a calibration `problem` using one or more iteration `controls`, from the package
IterationControls.jl. See the "Extended help" section of [`CalibrationProblem`](@ref) for
examples.

Not recommended for Gauss-Newton optimisers (`LevenbergMarquardt` or `Dogleg`).

"""
solve!(args...; kwargs...) = IterationControl.train!(args...; kwargs...)
solve!(problem::CurveOptimisationProblem) = IterationControl.train!(problem, 0)

"""
    loss(problem)

Return the sum of squares loss for a calibration `problem`, as constructed with
[`CalibrationProblem`](@ref).

"""
loss(args...; kwargs...) = IterationControl.loss(args...; kwargs...)
