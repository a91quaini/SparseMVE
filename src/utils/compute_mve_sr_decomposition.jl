using LinearAlgebra
using SparseMVE.UTILS: compute_sr, 
                       compute_mve_sr

"""
    compute_mve_sr_decomposition(
      μ, 
      Σ,
      μ_sample, 
      Σ_sample,
      max_card,
      sparse_mve;
      max_comb=0,
      do_checks=true
    ) -> NamedTuple

Perform the same Sharpe‐ratio decomposition as before, but call
`sparse_mve(μ_sample, Σ_sample, max_card; max_comb, do_checks)` so
you can swap in any cardinality‐constrained MVE routine.

# Arguments
- `μ::AbstractVector{<:Real}` population mean
- `Σ::AbstractMatrix{<:Real}` pupulation covariance 
- `μ_sample::AbstractVector{<:Real}` sample mean
- `Σ_sample::AbstractMatrix{<:Real}` sample covariance  
- `max_card::Integer`  
    maximum support size  
- `sparse_mve::Function` 
    A function of the form  
      `(μ::Vector, Σ::Matrix, max_card::Integer; max_comb, do_checks) -> NamedTuple(sr, weights, selection)`
- `max_comb::Integer=0`  
    if 0 ⇒ exhaustive search, else random trials  
- `do_checks::Bool=true`  
    whether to run input assertions  

# Returns
A named tuple with fields
- `sample_mve_sr`           : unconstrained MVE SR on the sample  
- `sample_sparse_mve_sr`     : best‐k SR on the sample (via `sparse_mve`)  
- `sparse_mve_sr_est_term`   : apply sample best‐k weights to population SR  
- `sparse_mve_sr_sel_term`   : best‐k SR on the population  
"""
function compute_mve_sr_decomposition(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    μ_sample::AbstractVector{<:Real},
    Σ_sample::AbstractMatrix{<:Real},
    max_card::Integer,
    sparse_mve::Function;
    max_comb::Integer = 0,
    do_checks::Bool  = false
)
    # basic checks
    n = length(μ)
    if do_checks
        @assert size(Σ)       == (n,n)          "pop Σ must be $n×$n"
        @assert length(μ_sample) == size(Σ_sample,1) == size(Σ_sample,2) "sample dims mismatch"
        @assert 1 ≤ max_card ≤ length(μ_sample) "max_card must ∈ 1…n_sample"
        @assert max_comb ≥ 0 "max_comb must be ≥ 0"
    end

    # 1) sample‐unconstrained MVE SR
    sample_mve_sr = compute_mve_sr(μ_sample, Σ_sample)

    # 2) sample‐best‐k via the user‐supplied sparse_mve
    sparse_mve = sparse_mve(μ_sample, Σ_sample, max_card;
                       max_comb=max_comb)

    # 3) estimation term: apply sample best‐k weights to population
    est_term = compute_sr(sparse_mve.weights, μ, Σ;
                         selection=sparse_mve.selection)

    # 4) selection term: best‐k on population
    sel_term = compute_mve_sr(μ, Σ;
                              selection=sparse_mve.selection)

    return (
      sample_mve_sr          = sample_mve_sr,
      sample_sparse_mve_sr   = sparse_mve.sr,
      sparse_mve_sr_est_term = est_term,
      sparse_mve_sr_sel_term = sel_term,
    )
end
