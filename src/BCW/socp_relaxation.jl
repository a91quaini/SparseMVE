using JuMP
using MosekTools
using LinearAlgebra

"""
QCQP-based SOCP relaxation for cardinality-constrained MV portfolios.
Returns `(Dual, socp_objective_value)`.
"""
function portfolios_socp(thePortfolio::SparsePortfolioData)
    n = thePortfolio.f      # dimension of α
    m = thePortfolio.m      # number of linear constraints
    f = thePortfolio.n      # portfolio asset count as second dimension in X

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_PFEAS", 1e-5)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", 1e-5)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)
    set_optimizer_attribute(model, "MSK_IPAR_MAX_NUM_WARNINGS", 0)

    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, t >= 0)
    @variable(model, v[1:f] >= 0)
    @variable(model, w[1:f])
    @variable(model, βl[1:m] >= 0)
    @variable(model, βu[1:m] >= 0)
    @variable(model, ρ[1:n] >= 0)

    # primal constraints
    @constraint(model, w .>= thePortfolio.X' * α
                         .+ thePortfolio.A' * (βl .- βu)
                         .+ λ .* ones(f)
                         .+ ρ
                         .- thePortfolio.d)

    # conic quadratic (epigraph) constraints: v_i + t + ρ_i*min_inv_i >= (γ_i/2)*w_i^2
    @constraint(model, [i=1:f], v[i] + t + ρ[i]*thePortfolio.min_investment[i] >= (thePortfolio.γ[i]/2) * w[i]^2)

    # objective: maximize dual
    expr = -0.5 * dot(α, α)
    expr += dot(thePortfolio.Y, α)
    expr += λ
    expr += dot(βl, thePortfolio.l)
    expr -= dot(βu, thePortfolio.u)
    expr -= sum(v)
    expr -= thePortfolio.k * t

    @objective(model, Max, expr)

    optimize!(model)
    status = termination_status(model)
    objval = objective_value(model)

    # build Dual
    dual = Dual(
      value.(α),
      value(λ),
      value.(βl),
      value.(βu),
      value.(ρ),
      value.(w),
      objval,
      status
    )

    return dual, objval
end

# (Optional) Full SOCP version, kept for reference
function portfolios_socp2(thePortfolio::SparsePortfolioData)
    n = size(thePortfolio.X, 1)
    m = size(thePortfolio.A, 1)
    f = size(thePortfolio.X, 2)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8)

    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, t >= 0)
    @variable(model, v[1:f] >= 0)
    @variable(model, w[1:f])
    @variable(model, βl[1:m] >= 0)
    @variable(model, βu[1:m] >= 0)
    @variable(model, ρ[1:n] >= 0)
    @variable(model, τ >= 0)

    @constraint(model, w .>= thePortfolio.X' * α
                         .+ thePortfolio.A' * (βl .- βu)
                         .+ λ .* ones(f)
                         .+ ρ
                         .- thePortfolio.d)

    @constraint(model, [i=1:f],
        v[i] + t + ρ[i]*thePortfolio.min_investment[i] + 1 >=
        norm([sqrt(2*thePortfolio.γ[i])*w[i]; v[i] + t + ρ[i]*thePortfolio.min_investment[i] - 1]))
    @constraint(model,
        τ + 1 >= norm([2α; τ - 1]))

    expr2 = -0.5*τ
    expr2 += dot(thePortfolio.Y, α)
    expr2 += λ
    expr2 += dot(βl, thePortfolio.l)
    expr2 -= dot(βu, thePortfolio.u)
    expr2 -= sum(v)
    expr2 -= thePortfolio.k * t
    @objective(model, Max, expr2)

    optimize!(model)
    status = termination_status(model)
    objval = objective_value(model)

    dual = Dual(
      value.(α),
      value(λ),
      value.(βl),
      value.(βu),
      value.(ρ),
      value.(w),
      status,
      objval
    )

    return dual, objval
end
