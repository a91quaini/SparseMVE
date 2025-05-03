using LinearAlgebra

"""
    compute_mve_sr(
        mu::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real};
        selection::Union{Nothing,AbstractVector{<:Integer}} = nothing
    )

Compute the maximum Sharpe ratio

    √( μ_S' * Σ_S^{-1} * μ_S )

over either the full universe or a selected subset of assets.

# Arguments
- `μ`         length-n expected-returns vector  
- `Σ`         n×n covariance matrix (must be symmetric positive-definite)  
- `selection` (optional) 1-based indices of assets  

# Returns
- `sr::T`    the resulting Sharpe ratio  

Throws an `AssertionError` if dimensions mismatch or `selection` out of bounds.
"""
function compute_mve_sr(
    mu::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::Union{Nothing,AbstractVector{<:Integer}} = nothing
)
    n = length(mu)
    @assert size(Σ) == (n,n) "Σ must be an $n×$n matrix, got $(size(Σ))"

    if selection === nothing
        # full‐universe: one factorization and solve
        F = cholesky(Σ; check = true)
        x = F \ mu
        return sqrt(dot(mu, x))
    else
        idx = selection
        @assert all(1 .<= idx .<= n) "selection indices out of bounds (1:$n)"
        # build subproblem on S = idx
        μ_S = @view mu[idx]
        Σ_S = @view Σ[idx, idx]
        F   = cholesky(Σ_S; check = true)
        x   = F \ μ_S
        return sqrt(dot(μ_S, x))
    end
end