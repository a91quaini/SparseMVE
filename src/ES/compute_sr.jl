using LinearAlgebra

"""
    compute_sr(w, μ, Σ; selection=nothing)

Compute the Sharpe ratio

Sharpe = wᵀ μ / √(wᵀ Σ w)

where: 

- `w`         is a length-n weight vector,
- `μ`         is a length-n expected-returns vector,
- `Σ`         is the n×n covariance matrix,
- `selection` (optional) is a vector of 1-based indices selecting a subset of assets.

Throws an `AssertionError` if dimensions mismatch or if `selection` is out of range.
"""

function compute_sr(
    w::AbstractVector{<:Real},
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::Union{Nothing,AbstractVector{<:Integer}} = nothing
)
    n = length(w)
    @assert length(μ) == n       "length(w) = $n but length(μ) = $(length(μ))"
    @assert size(Σ) == (n,n)     "Σ must be an $n×$n matrix, got size $(size(Σ))"
    idx = selection === nothing ? axes(w,1) : selection
    @assert all(1 .<= idx .<= n) "selection indices out of bounds (1:$n)"

    # subset
    w_sel = view(w, idx)
    μ_sel = view(μ, idx)
    Σ_sel = view(Σ, idx, idx)

    num = dot(w_sel, μ_sel)
    den = sqrt(dot(w_sel, Σ_sel * w_sel))
    return num / den
end
