using LinearAlgebra

"""
    compute_mve_sr(mu, Σ; selection=nothing)

Compute the maximum achievable Sharpe ratio (mean-variance efficient) for
either the full universe or a selected subset of assets.

# Arguments
- `mu::AbstractVector{<:Real}`: expected returns (length n)
- `Σ::AbstractMatrix{<:Real}`: covariance matrix (n×n)
- `selection::Union{Nothing,AbstractVector{<:Integer}}`: optional 1-based indices of assets

# Returns
- `sr::Float64`: the optimal Sharpe ratio, i.e.

```text
    sr = sqrt(mu_S' * Σ_S^{-1} * mu_S)
```

where `mu_S` and `Σ_S` are the (sub)vectors/matrix restricted to `selection`,
or the full universe if `selection===nothing`.

Throws an `AssertionError` if dimensions mismatch or indices out of bounds.
"""
function compute_mve_sr(
    mu::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::Union{Nothing,AbstractVector{<:Integer}} = nothing
)
    n = length(mu)
    @assert size(Σ) == (n,n) "Σ must be an $n×$n matrix, got $(size(Σ))"

    idx = selection === nothing ? axes(mu,1) : selection
    @assert all(1 .<= idx .<= n) "selection indices out of bounds (1:$n)"

    # subset
    mu_S = view(mu, idx)
    Σ_S  = view(Σ, idx, idx)

    # solve Σ_S * x = mu_S, then sr = sqrt(mu_S' * x)
    x = Σ_S \ mu_S
    return sqrt(dot(mu_S, x))
end
