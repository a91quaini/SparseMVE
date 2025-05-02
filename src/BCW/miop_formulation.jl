using JuMP
using MosekTools
using LinearAlgebra
using Random

"""
Cutting-plane algorithm for cardinality-constrained MV portfolios.
"""
function cutting_planes_portfolios(
    thePortfolio::SparsePortfolioData,
    ΔT_max::Float64 = 600.0,
    gap::Float64    = 1e-4;
    numRandomRestarts::Int       = 5,
    TrackCuts::Int               = 0,
    useSOCPLB::Bool              = false,
    minReturnConstraint::Bool    = false,
    useHeuristic::Bool           = true,
    useWarmStart::Bool           = true,
    useCopyOfVariables::Bool     = false,
    usingKelleyPrimal::Bool      = false,
    maxCutCallbacks::Int         = 0
)
    numIters = 0
    bestbound = -Inf

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_MAX_TIME", ΔT_max)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", gap)

    # decision variables
    @variable(model, s[1:thePortfolio.n], Bin)
    @variable(model, t >= -1e12)
    @objective(model, Min, t)

    # cardinality constraints
    @constraint(model, sum(s) <= thePortfolio.k)
    @constraint(model, sum(s) >= 1)i

    # minimum return option
    if minReturnConstraint
        @constraint(model,
            sum(s[i] * (thePortfolio.A[1,i] > thePortfolio.l[1]) for i in 1:thePortfolio.n) >= 1.0)
    end

    # optional warm-start variables
    if useCopyOfVariables
        @variable(model, x_master[1:thePortfolio.n] >= 0)
        @constraint(model, thePortfolio.A * x_master .>= thePortfolio.l)
        @constraint(model, thePortfolio.A * x_master .<= thePortfolio.u)
        @constraint(model, [i=1:thePortfolio.n], x_master[i] >= s[i] * thePortfolio.min_investment[i])
        @constraint(model, [i=1:thePortfolio.n], x_master[i] <= thePortfolio.u[i+1] * s[i])
        @constraint(model, sum(x_master) == 1.0)
    end

    # buy-in constraint
    @constraint(model,
        sum(s[i] * thePortfolio.min_investment[i] for i in 1:thePortfolio.n) <= 1.0)

    # optional SOCP lower bound
    if useSOCPLB
        dual_vars, socp_lb = portfolios_socp(thePortfolio)
        bestbound = socp_lb
    end

    # optional warm start
    bests0 = zeros(thePortfolio.n)
    if useWarmStart
        bests0 = getWarmStart(thePortfolio, numRandomRestarts)
        cut0 = portfolios_objective(thePortfolio, bests0)
        if sum(bests0) <= thePortfolio.k && cut0.status == :Optimal
            set_start_value.(s, bests0)
            @constraint(model, t >= cut0.p + dot(cut0.∇s, s .- bests0))
        end
    end

    # solve
    optimize!(model)

    # extract solution
    s_opt = round.(Int, clamp.(value.(s), 0, 1))
    indices = findall(x->x>0, s_opt)
    dual_vars = inner_dual(thePortfolio, indices)
    w = thePortfolio.γ[indices] .* dual_vars.w

    return CardinalityConstrainedPortfolio(
        indices,
        w,
        dual_vars.λ,
        dual_vars.α,
        dual_vars.βl,
        dual_vars.βu,
        dual_vars.ρ,
        (objective_value(model) - bestbound) / abs(objective_value(model)),
        numIters,
        (thePortfolio.μ[indices]' * w)[1],
        (w' * thePortfolio.X[:,indices]' * thePortfolio.X[:,indices] * w)[1],
        missing,
        missing
    )
end

# Portfolio evaluations & warm start helpers
function portfolios_objective(thePortfolio::SparsePortfolioData, s::Vector{Float64})
    indices = findall(x->x>0.5, s)
    dual_vars = inner_dual(thePortfolio, indices)
    p = dual_vars.ofv
    ρ_full = zeros(thePortfolio.n)
    ρ_full[indices] .= dual_vars.ρ
    w = thePortfolio.X' * dual_vars.α .+
        dual_vars.λ .* ones(thePortfolio.n) .+
        thePortfolio.A' * (dual_vars.βl .- dual_vars.βu) .+
        ρ_full .- thePortfolio.d
    for i in eachindex(w)
        if abs(s[i]) < 1e-8 && w[i] <= -1e-8
            w[i] = 0.0
        end
    end
    ∇s = -thePortfolio.γ .* (w.^2) ./ 2 .+ ρ_full .* thePortfolio.min_investment
    return Cut(p[1], ∇s, dual_vars.status)
end

function portfolios_objective2(
    thePortfolio::SparsePortfolioData,
    s0::Vector{Float64}
)
    indices = findall(x->x>1e-6, s0)
    dual_vars = inner_dual2(thePortfolio, indices, s0)
    p = dual_vars.ofv
    ρ_full = zeros(thePortfolio.n)
    ρ_full[indices] .= dual_vars.ρ
    w = thePortfolio.X' * dual_vars.α .+
        dual_vars.λ .* ones(thePortfolio.n) .+
        thePortfolio.A' * (dual_vars.βl .- dual_vars.βu) .+
        ρ_full .- thePortfolio.d
    for i in eachindex(w)
        if abs(s0[i]) < 1e-8 && w[i] <= -1e-8
            w[i] = 0.0
        end
    end
    ∇s = -thePortfolio.γ .* (w.^2) ./ 2 .+ ρ_full .* thePortfolio.min_investment
    return Cut(p[1], ∇s, dual_vars.status)
end

function getWarmStart(
    thePortfolio::SparsePortfolioData,
    numRandomRestarts::Int
)
    bestp0 = Inf
    bests0 = zeros(thePortfolio.n)
    for _ in 1:numRandomRestarts
        idx0 = portfolios_hillclimb(thePortfolio,
                                    sort(shuffle(1:thePortfolio.n)[1:thePortfolio.k]))[1]
        while sum(thePortfolio.min_investment[idx0]) > 1.0
            idx0 = filter(e->e != rand(idx0), idx0)
        end
        s0 = zeros(thePortfolio.n)
        s0[idx0] .= 1
        cut = portfolios_objective(thePortfolio, s0)
        if cut.p < bestp0
            bestp0 = cut.p
            bests0 = s0
        end
    end
    return bests0
end
