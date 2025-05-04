using LinearAlgebra
using Statistics
using Distributions
using SparseMVE.UTILS: compute_mve_sr_decomposition

export simulate_mve_sr

"""
    simulate_mve_sr(
      μ,
      Σ,
      n_obs,
      max_card,
      sparse_mve;
      max_comb=0,
      do_checks=true
    ) -> NamedTuple

Simulate `n_obs` IID draws from 𝐍(μ,Σ), fit the sample mean & covariance,
then delegate to `compute_mve_sr_decomposition` with the user‐supplied
`sparse_mve` function.

# Arguments
- `μ::AbstractVector{<:Real}`         : population mean (length n)
- `Σ::AbstractMatrix{<:Real}`         : population covariance (n×n)
- `n_obs::Integer`                    : number of simulated observations (≥1)
- `max_card::Integer`                 : max support size (1…n)
- `sparse_mve::Function`              : a cardinality‐constrained MVE routine,
    signature `(μ, Σ, max_card; max_comb, do_checks) -> NamedTuple(sr, weights, selection)`
- `max_comb::Integer=0`               : if 0 ⇒ exhaustive, else # random supports per k
- `do_checks::Bool=true`              : whether to run input assertions

# Returns
The same named tuple returned by `compute_mve_sr_decomposition`, with fields
- `sample_mve_sr`
- `sample_sparse_mve_sr`
- `mve_sparse_sr_est_term`
- `mve_sparse_sr_sel_term`
"""
function simulate_mve_sr(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    n_obs::Integer,
    max_card::Integer,
    sparse_mve::Function;
    max_comb::Integer = 0,
    do_checks::Bool  = true
)
    n = length(μ)
    if do_checks
        @assert n_obs ≥ 1                        "n_obs must be ≥ 1"
        @assert size(Σ) == (n,n)                "Σ must be an $n×n matrix"
        @assert 1 ≤ max_card ≤ n                "max_card must be between 1 and $n"
        @assert max_comb ≥ 0                    "max_comb must be non-negative"
    end

    # simulate n_obs samples from N(μ, Σ)
    mvn   = MvNormal(μ, Σ)
    draws = rand(mvn, n_obs)                   # n × n_obs

    # compute sample moments
    μ_sample = vec(mean(draws; dims=2))        # length-n
    Σ_sample = cov(draws; dims=2)              # n×n

    # perform the decomposition using the provided sparse_mve routine
    return compute_mve_sr_decomposition(
        μ, Σ,
        μ_sample, Σ_sample,
        max_card, sparse_mve;
        max_comb = max_comb,
        do_checks = do_checks
    )
end
