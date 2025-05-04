using LinearAlgebra

"""
    compute_mve_weights(
        μ::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real};
        selection::Union{Nothing,AbstractVector{<:Integer}} = nothing,
        gamma::Real = 1.0,
        do_checks::Bool = false
    ) -> Vector{Float64}

Compute the weights of the mean–variance‐efficient portfolio (no cardinality constraint)
over either the full universe or a specified subset.

# Arguments
- `μ` : expected‐return vector, length n
- `Σ` : n×n covariance matrix (SPD)
- `selection` : optional 1-based indices of a subset; if `nothing`, uses full universe
- `gamma` : risk‐aversion parameter (must be > 0)
- `do_checks` : whether to run input checks

# Returns
- length‐n vector of weights `w`, with zeros off‐support.
"""
function compute_mve_weights(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    gamma::Real = 1.0,
    do_checks::Bool = false
)::Vector{Float64}
    n = length(μ)
    if do_checks
        @assert size(Σ) == (n, n)        "Σ must be an $n×$n matrix, got size $(size(Σ))"
        @assert gamma > 0.0              "gamma must be positive, got $gamma"
        if selection !== nothing
            idx = selection
            @assert all(1 .<= idx .<= n) "selection indices out of bounds (1:$n)"
        end
    end

    if selection === nothing || length(selection) == n
        # full-universe solution
        F = cholesky(Σ; check=true)
        w = F \ μ
        w ./= gamma
        return w
    else
        # restricted to `selection`
        idx = selection
        μS = @view μ[idx]
        ΣS = @view Σ[idx, idx]
        F  = cholesky(ΣS; check=true)
        wS = F \ μS
        w = zeros(Float64, n)
        w[idx] .= wS ./ gamma
        return w
    end
end

export compute_mve_weights
