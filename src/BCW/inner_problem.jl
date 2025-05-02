using JuMP
using MosekTools
using LinearAlgebra

"""
  inner_dual(thePortfolio, indices)

Solve the dual QP for a fixed support `indices` via JuMP+Mosek.
Returns a `Dual(α, λ, βl, βu, ρ, w, ofv, status)`.
"""
function inner_dual(thePortfolio::SparsePortfolioData, indices::Vector{Int})
    n = thePortfolio.f
    m = thePortfolio.m
    f = length(indices)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_PFEAS", 1e-6)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", 1e-6)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    set_optimizer_attribute(model, "MSK_IPAR_MAX_NUM_WARNINGS", 0)

    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, w[1:f])
    @variable(model, ρ[1:f] ≥ 0)
    @variable(model, βl[1:m] ≥ 0)
    @variable(model, βu[1:m] ≥ 0)

    # primal feasibility constraints
    @constraint(model, w .>= thePortfolio.X[:, indices]' * α
                         .+ thePortfolio.A[:, indices]' * (βl .- βu)
                         .+ λ .* ones(f)
                         .+ ρ
                         .- thePortfolio.d[indices])

    # objective
    expr = -0.5 * dot(α, α)
    expr -= sum((thePortfolio.γ[indices] ./ 2) .* w .* w)
    expr += dot(thePortfolio.Y, α)
    expr += λ
    expr += dot(βl, thePortfolio.l)
    expr -= dot(βu, thePortfolio.u)
    expr += dot(ρ, thePortfolio.min_investment[indices])

    @objective(model, Max, expr)

    optimize!(model)
    status = termination_status(model)

    return Dual(
        value.(α),
        value(λ),
        value.(βl),
        value.(βu),
        value.(ρ),
        value.(w),
        objective_value(model),
        status,
    )
end


"""
  inner_dual2(thePortfolio, indices, s)

Variant of `inner_dual` with scaling vector `s`.
"""
function inner_dual2(thePortfolio::SparsePortfolioData, indices::Vector{Int}, s::Vector{Float64})
    n = thePortfolio.f
    m = thePortfolio.m
    f = length(indices)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_PFEAS", 1e-6)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", 1e-6)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    set_optimizer_attribute(model, "MSK_IPAR_MAX_NUM_WARNINGS", 0)

    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, w[1:f])
    @variable(model, ρ[1:f] ≥ 0)
    @variable(model, βl[1:m] ≥ 0)
    @variable(model, βu[1:m] ≥ 0)

    @constraint(model, w .>= thePortfolio.X[:, indices]' * α
                         .+ thePortfolio.A[:, indices]' * (βl .- βu)
                         .+ λ .* ones(f)
                         .+ ρ
                         .- thePortfolio.d[indices])

    expr = -0.5 * dot(α, α)
    expr -= sum((thePortfolio.γ[indices] ./ 2) .* s[indices] .* w .* w)
    expr += dot(thePortfolio.Y, α)
    expr += λ
    expr += dot(βl, thePortfolio.l)
    expr -= dot(βu, thePortfolio.u)
    expr += dot(ρ, s[indices] .* thePortfolio.min_investment[indices])

    @objective(model, Max, expr)

    optimize!(model)
    status = termination_status(model)

    return Dual(
        value.(α),
        value(λ),
        value.(βl),
        value.(βu),
        value.(ρ),
        value.(w),
        objective_value(model),
        status,
    )
end
