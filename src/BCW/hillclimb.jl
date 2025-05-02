using JuMP
using MosekTools
using LinearAlgebra  # for vector/matrix utilities


"""
Hill-climbing heuristic for cardinality-constrained mean-variance portfolios.
Replaces CPLEX with Mosek via JuMP/MosekTools.
Compatible with Julia 1.11+.
"""
function portfolios_hillclimb(thePortfolio::SparsePortfolioData, indices_new::Vector{Int})
    L = 1.0e1                   # step parameter; tune if needed
    λ = 0.0
    iter = 0
    indices = Int[]
    α  = zeros(thePortfolio.f)
    βl = zeros(size(thePortfolio.A, 1))
    βu = similar(βl)
    ρ  = zeros(thePortfolio.n)
    SoR = 1.7                  # smoothing coefficient

    while (iter < 20 || (indices != indices_new && iter <= 50))
        iter += 1
        indices = copy(indices_new)

        # Solve the dual subproblem using Mosek (inner_dual must use MosekTools)
        dual_vars = inner_dual(thePortfolio, indices)

        # full rho vector
        ρ_full = zeros(thePortfolio.n)
        ρ_full[indices] .= dual_vars.ρ

        # update averaged dual estimates
        α  .= (α*(iter - SoR) .+ SoR*dual_vars.α)  ./ iter
        λ   = (λ*(iter - SoR) + SoR*dual_vars.λ)   /  iter
        βl .= (βl*(iter - SoR) .+ SoR*dual_vars.βl) ./ iter
        βu .= (βu*(iter - SoR) .+ SoR*dual_vars.βu) ./ iter
        ρ  .= (ρ*(iter - SoR)  .+ SoR*ρ_full)       ./ iter

        # primal heuristic update
        p = (-(thePortfolio.γ ./ 2.0)) .* ((α' * thePortfolio.X .+
                                           (βl .- βu)' * thePortfolio.A .+
                                           λ .+ ρ .- thePortfolio.d).^2) .+
            (thePortfolio.min_investment .* ρ)'

        w = thePortfolio.γ[indices] .* (thePortfolio.X[:, indices]' * α .+
                                        thePortfolio.A[:, indices]' * (βl .- βu) .+
                                        λ .+ ρ[indices] .- thePortfolio.d[indices])

        # form full weight vector and select top-k moves
        x = zeros(thePortfolio.n)
        x[indices] .= w
        deviations = abs.(x .- (1/L) .* p)
        new_idx = sortperm(deviations, rev=true)[1:thePortfolio.k]
        indices_new = sort(new_idx)

        # ensure feasibility under min_investment
        while sum(thePortfolio.min_investment[indices_new]) > 1.0
            drop = rand(indices_new)
            indices_new = setdiff(indices_new, [drop])
        end
    end

    # final solve
    final_dual = inner_dual(thePortfolio, indices_new)
    return indices_new, thePortfolio.γ .* final_dual.w
end
