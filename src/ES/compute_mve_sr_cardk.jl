using LinearAlgebra
using Combinatorics 
using SparseMVE.ES: compute_mve_sr

"""
    compute_mve_sr_cardk(
      μ, Σ, max_card;
      max_comb=0,
      gamma=0.0,
      do_checks=true
    ) -> (sr, weights, selection)

Find the maximum‐Sharpe‐ratio portfolio over all supports of size
1…`max_card`.  If `max_comb==0` it exhaustively enumerates each
k‐subset; otherwise it draws `max_comb` random supports of each size.

# Arguments
- `μ::AbstractVector{<:Real}`   : expected‐return vector (length n)  
- `Σ::AbstractMatrix{<:Real}`   : n×n covariance, must match μ  
- `max_card::Integer`           : maximum number of nonzeros (1…n)  
- `max_comb::Integer=0`         : if 0 ⇒ exhaustive search; else random sampling  
- `gamma::Real=0.0`             : risk-aversion scalar (0 means pure ratio problem)  
- `do_checks::Bool=true`        : perform input assertions  

# Returns
A named tuple `(sr, weights, selection)` where  
- `sr::Float64`        is the best Sharpe ratio found  
- `weights::Vector{Float64}` is the length-n weight vector (zeros off-support)  
- `selection::Vector{Int}`   are the 1-based indices of the selected assets  
"""
function compute_mve_sr_cardk(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    max_card::Integer;
    max_comb::Integer=0,
    gamma::Real=0.0,
    do_checks::Bool=true
)
    n = length(μ)
    if do_checks
        @assert size(Σ) == (n,n)    "Σ must be an $n×n matrix"
        @assert 1 ≤ max_card ≤ n    "max_card must be between 1 and $n"
        @assert max_comb ≥ 0        "max_comb must be non-negative"
        @assert gamma ≥ 0           "gamma must be non-negative"
    end

    best_sr  = -Inf
    best_sel = Int[]

    if max_comb == 0
        # exhaustive search
        for k in 1:max_card, sel in combinations(1:n, k)
            sr = compute_mve_sr(μ, Σ; selection=sel)
            if sr > best_sr
                best_sr, best_sel = sr, sel
            end
        end
    else
        # random sampling
        for k in 1:max_card
            for _ in 1:max_comb
                sel = randperm(n)[1:k]
                sr  = compute_mve_sr(μ, Σ; selection=sel)
                if sr > best_sr
                    best_sr, best_sel = sr, sel
                end
            end
        end
    end

    # now build final weights on the chosen support
    μS = view(μ, best_sel)
    ΣS = view(Σ, best_sel, best_sel)
    wS = ΣS \ μS
    # only apply risk-aversion scaling if gamma>0
    if gamma > 0
        wS ./= gamma
    end

    w = zeros(eltype(μ), n)
    w[best_sel] .= wS

    return (sr=best_sr, weights=w, selection=best_sel)
end
