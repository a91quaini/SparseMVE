using JuMP
using MosekTools
using LinearAlgebra  # for numerical utilities
using Random          # for shuffle in warm start

"""
Hill-climbing heuristic for cardinality-constrained mean-variance portfolios.
Returns a tuple `(indices, weights)` of length-(k) indices and length-(n) weights.
"""
function portfolios_hillclimb(thePortfolio::SparsePortfolioData, indices_new::Vector{Int})
    L = 1e1                   # step parameter; tune if needed
    λ = 0.0
    iter = 0
    indices = Int[]
    α  = zeros(thePortfolio.f)
    βl = zeros(thePortfolio.m)
    βu = similar(βl)
    ρ  = zeros(thePortfolio.n)
    SoR = 1.7                 # smoothing coefficient
    n = thePortfolio.n        # number of assets

    while (iter < 20 || (indices != indices_new && iter <= 50))
        iter += 1
        indices = copy(indices_new)

        # dual solve
        dual_vars = inner_dual(thePortfolio, indices)
        ρ_full = zeros(n)
        ρ_full[indices] .= dual_vars.ρ

        # update dual averages
        α  .= (α*(iter - SoR) .+ SoR*dual_vars.α)  ./ iter
        λ   = (λ*(iter - SoR) + SoR*dual_vars.λ)   /  iter
        βl .= (βl*(iter - SoR) .+ SoR*dual_vars.βl) ./ iter
        βu .= (βu*(iter - SoR) .+ SoR*dual_vars.βu) ./ iter
        ρ  .= (ρ*(iter - SoR)  .+ SoR*ρ_full)       ./ iter

        # primal heuristic: compute p as 1D vector length n
        row = thePortfolio.X' * α .+ thePortfolio.A' * (βl .- βu) .+ λ .+ ρ .- thePortfolio.d
        p   = -(thePortfolio.γ ./ 2) .* (row .^ 2) .+ (thePortfolio.min_investment .* ρ)

        # compute tentative weights on support
        w_support = thePortfolio.γ[indices] .* (
            thePortfolio.X[:, indices]' * α .+ thePortfolio.A[:, indices]' * (βl .- βu) .+ λ .+ ρ[indices] .- thePortfolio.d[indices]
        )

        # full-vector x for deviations
        x = zeros(n)
        x[indices] .= w_support

        # pick top-k by deviation
        deviations = abs.(x .- (1/L) .* p)
        new_idx   = sortperm(deviations, rev=true)[1:thePortfolio.k]
        indices_new = sort(new_idx)

        # enforce min_investment feasibility
        while sum(thePortfolio.min_investment[indices_new]) > 1.0
            drop = rand(indices_new)
            indices_new = filter(e -> e != drop, indices_new)
        end
    end

    # build final weights vector of length n
    final_dual = inner_dual(thePortfolio, indices_new)
    weights    = zeros(n)
    weights[indices_new] .= thePortfolio.γ[indices_new] .* final_dual.w
    return indices_new, weights
end
