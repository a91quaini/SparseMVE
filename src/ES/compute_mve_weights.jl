using LinearAlgebra

"""
    compute_mve_weights(
        μ::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real};
        selection::Union{Nothing,AbstractVector{<:Integer}} = nothing,
        gamma::Real = 1.0,
        do_checks::Bool = true
    ) -> Vector{Float64}

    Compute the weights of the Maximum Variance Efficiency (MVE) portfolio for a given set of assets.

    The weights are computed as:

        w = Σ_S^{-1} * μ_S / γ

    where `μ_S` is the expected returns vector, `Σ_S` is the covariance matrix of the selected subset of assets, and `γ` is the risk-aversion parameter.

    # Arguments
    - `μ::AbstractVector{<:Real}`: A length-`n` vector of expected returns for the assets.
    - `Σ::AbstractMatrix{<:Real}`: An `n×n` covariance matrix of the assets. Must be symmetric positive-definite.
    - `selection::Union{Nothing,AbstractVector{<:Integer}}`: (Optional) A vector of 1-based indices specifying a subset of assets to consider. If `nothing`, the full universe of assets is used.
    - `gamma::Real`: The risk-aversion parameter. Must be positive. Default is `1.0`.
    - `do_checks::Bool`: Whether to perform input validation checks. Default is `true`.

    # Returns
    - `Vector{Float64}`: A length-`n` vector of portfolio weights.

    # Throws
    - `AssertionError`: If the dimensions of `μ` and `Σ` do not match.
    - `AssertionError`: If `gamma` is not positive.
    - `AssertionError`: If the `selection` indices are out of bounds.


"""

function compute_mve_weights(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    gamma::Real = 1.0,
    do_checks::Bool = true
)
    n = length(μ)

    if do_checks
        @assert size(Σ) == (n,n)         "Σ must be an $n×n matrix, got size $(size(Σ))"
        @assert gamma > 0               "gamma must be positive, got $gamma"
        if selection !== nothing
            idx = selection
            @assert all(1 .<= idx .<= n) "selection indices out of bounds (1:$n)"
        end
    end

    if selection === nothing || length(selection) == n
        F = cholesky(Σ; check = true)
        w = F \ μ
        return w ./ gamma
    end

    idx = selection
    μS = @view μ[idx]
    ΣS = @view Σ[idx, idx]
    F  = cholesky(ΣS; check = true)
    wS = F \ μS

    w = zeros(Float64, n)
    w[idx] .= wS ./ gamma
    return w
end

