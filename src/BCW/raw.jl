using JuMP
using MosekTools
using LinearAlgebra
using MathOptInterface

const MOI = MathOptInterface

"""
Mixed‐integer Big‐M QP for cardinality‐constrained MV portfolios using Mosek.
Returns a `CardinalityConstrainedPortfolio` with primal solution.
"""
function mosek_raw_bigM(
    thePortfolio::SparsePortfolioData,
    ΔT_max::Float64 = 3600.0,
    gap::Float64     = 1e-4,
    debugLevel::Int  = 0
)
    n, m, k = thePortfolio.n, thePortfolio.m, thePortfolio.k
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_MAX_TIME", ΔT_max)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", gap)

    @variable(model, z[1:n], Bin)
    @variable(model, x[1:n] >= 0)

    @constraint(model, sum(z) <= k)
    @constraint(model, [i=1:n], x[i] <= z[i])
    @constraint(model, [i=1:n], x[i] >= z[i] * thePortfolio.min_investment[i])
    @constraint(model, sum(x) == 1.0)
    @constraint(model, thePortfolio.A * x .>= thePortfolio.l)
    @constraint(model, thePortfolio.A * x .<= thePortfolio.u)

    expr = 0.5 * sum(x[i]^2 / thePortfolio.γ[i] for i in 1:n)
    expr += 0.5 * dot(x, thePortfolio.X' * (thePortfolio.X * x))
    expr -= dot(thePortfolio.μ, x)
    expr += 0.5 * dot(thePortfolio.Y, thePortfolio.Y)
    @objective(model, Min, expr)

    optimize!(model)

    xopt = value.(x)
    zopt = value.(z)
    indices = findall(i -> zopt[i] > 0.5, 1:n)

    relgap  = MOI.get(model, MOI.RelativeGap())
    soltime = MOI.get(model, MOI.SolveTime())
    nodecnt = MOI.get(model, MOI.NodeCount())

    return CardinalityConstrainedPortfolio(
        indices,
        xopt[indices],
        0.0,
        zeros(n),
        zeros(m),
        zeros(m),
        zeros(n),
        relgap,
        relgap,
        0,
        dot(thePortfolio.μ, xopt),
        dot(xopt, thePortfolio.X' * (thePortfolio.X * xopt)),
        soltime,
        nodecnt
    )
end

"""
Mixed‐integer SOC‐QP formulation via Mosek.
"""
function mosek_raw_MISOCP(
    thePortfolio::SparsePortfolioData,
    ΔT_max::Float64 = 3600.0,
    gap::Float64     = 1e-4,
    debugLevel::Int  = 0
)
    n, m, k = thePortfolio.n, thePortfolio.m, thePortfolio.k
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_MAX_TIME", ΔT_max)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", gap)

    @variable(model, z[1:n], Bin)
    @variable(model, x[1:n] >= 0)
    @variable(model, θ[1:n] >= 0)
    @variable(model, τ >= 0)

    @constraint(model, [i=1:n], θ[i] + z[i] >= norm([2.0 * x[i]; θ[i] - z[i]]))
    @constraint(model, sum(z) <= k)
    @constraint(model, thePortfolio.A * x .>= thePortfolio.l)
    @constraint(model, thePortfolio.A * x .<= thePortfolio.u)
    @constraint(model, [i=1:n], x[i] >= z[i] * thePortfolio.min_investment[i])
    @constraint(model, sum(x) == 1.0)
    @constraint(model, τ + 1.0 >= norm([2.0 * (thePortfolio.X * x); τ - 1.0]))

    expr = 0.5 * sum(θ[i] / thePortfolio.γ[i] for i in 1:n)
    expr += 0.5 * τ
    expr -= dot(thePortfolio.μ, x)
    expr += 0.5 * dot(thePortfolio.Y, thePortfolio.Y)
    @objective(model, Min, expr)

    optimize!(model)

    xopt = value.(x)
    zopt = value.(z)
    indices = findall(i -> zopt[i] > 0.5, 1:n)

    relgap  = MOI.get(model, MOI.RelativeGap())
    soltime = MOI.get(model, MOI.SolveTime())
    nodecnt = MOI.get(model, MOI.NodeCount())

    return CardinalityConstrainedPortfolio(
        indices,
        xopt[indices],
        0.0,
        zeros(n),
        zeros(m),
        zeros(m),
        zeros(n),
        relgap,
        relgap,
        0,
        dot(thePortfolio.μ, xopt),
        dot(xopt, thePortfolio.X' * (thePortfolio.X * xopt)),
        soltime,
        nodecnt
    )
end

"""
Continuous MISOCP relaxation via Mosek returns fractional z.
"""
function mosek_MISOCP_relaxation(
    thePortfolio::SparsePortfolioData,
    ΔT_max::Float64 = 3600.0
)
    n, k = thePortfolio.n, thePortfolio.k
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_MAX_TIME", ΔT_max)

    @variable(model, 0 <= z[1:n] <= 1)
    @constraint(model, sum(z) <= k)

    @objective(model, Min, 0.0)
    optimize!(model)

    return value.(z)
end

