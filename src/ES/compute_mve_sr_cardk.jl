using Combinatorics        # for combinations
import ..ES: compute_mve_sr

"""
    compute_mve_sr_cardk(
      μ, Σ, max_card;
      max_comb=0,
      gamma=0.0,
      do_checks=true
    ) -> NamedTuple{(:sr,:weights,:selection),Tuple{Float64,Vector{Float64},Vector{Int}}}

Find the maximum Sharpe‐ratio MVE portfolio over all supports of size
1…`max_card`.  If `max_comb==0`, exhaustively enumerates every
k‐subset; otherwise draws `max_comb` random supports for each
cardinality.

Arguments:

- `μ::AbstractVector{<:Real}`         : expected‐return vector, length n  
- `Σ::AbstractMatrix{<:Real}`         : n×n covariance, must match μ  
- `max_card::Integer`                 : maximum support size (1…n)  
- `max_comb::Integer=0`               : if 0 ⇒ exhaustive; else random‐sample this many subsets per k  
- `gamma::Real=0.0`                   : ridge penalty on Σ (Σ + γI)  
- `do_checks::Bool=true`              : run input assertions  

Returns `(sr, weights, selection)`:

- `sr::Float64`        : best Sharpe ratio found  
- `weights::Vector{Float64}` : full‐length weight vector (zeros off‐support)  
- `selection::Vector{Int}`   : 1‐based indices of the selected assets  
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
        @assert ndims(Σ)==2 && size(Σ,1)==n && size(Σ,2)==n   "Σ must be an $n×n matrix"
        @assert 1 ≤ max_card ≤ n                             "max_card must be between 1 and $n"
        @assert max_comb ≥ 0                                 "max_comb must be non‐negative"
        @assert gamma ≥ 0.0                                  "gamma must be non‐negative"
    end

    best_sr = -Inf
    best_sel = Int[]

    if max_comb == 0
        # exhaustive enumeration
        for k in 1:max_card
            for sel in combinations(1:n, k)
                sr = compute_mve_sr(μ, Σ; selection=sel)
                if sr > best_sr
                    best_sr, best_sel = sr, sel
                end
            end
        end
    else
        # random sampling of supports
        for k in 1:max_card
            for _ in 1:max_comb
                sel = randperm(n)[1:k]
                sr = compute_mve_sr(μ, Σ; selection=sel)
                if sr > best_sr
                    best_sr, best_sel = sr, sel
                end
            end
        end
    end

    # once we have best_sel, form Σ_S and μ_S
    k = length(best_sel)
    ΣS = @view Σ[best_sel, best_sel]
    μS = @view μ[best_sel]

    # include ridge penalty γI
    if gamma > 0
        ΣS = ΣS .+ gamma*I(k)
    end

    # compute weights on support: Σ_S⁻¹ μ_S
    wS = ΣS \ μS

    # embed into full‐length vector
    w = zeros(eltype(μ), n)
    w[best_sel] .= wS

    return (sr=best_sr, weights=w, selection=best_sel)
end