using Test
using LinearAlgebra
using Random
using Distributions

# bring in the routines under test
using SparseMVE.UTILS: simulate_mve_sr, compute_mve_sr_decomposition
using SparseMVE.ES: compute_sparse_mve_ES

@testset "simulate_mve_sr signature" begin
    @test isdefined(SparseMVE.UTILS, :simulate_mve_sr)
    @test isa(simulate_mve_sr, Function)
end

@testset "simulate_mve_sr reproducibility & functionality" begin
    # fix the RNG for reproducibility
    Random.seed!(1234)
    μ2 = [1.0, 2.0]
    Σ2 = Matrix{Float64}(I, 2, 2)

    # parameters
    n_obs    = 50
    max_card = 1

    # draw once manually
    mvn = MvNormal(μ2, Σ2)
    Random.seed!(1234)
    draws = rand(mvn, n_obs)             # 2×n_obs

    # sample moments
    μ_sample = vec(mean(draws; dims=2))
    Σ_sample = cov(draws; dims=2)

    # expected decomposition via compute_mve_sr_decomposition
    expected = compute_mve_sr_decomposition(
        μ2, Σ2,
        μ_sample, Σ_sample,
        max_card, compute_sparse_mve_ES;
        max_comb = 0,
        do_checks = true
    )

    # now run simulate_mve_sr (it reseeds internally to same seed)
    Random.seed!(1234)
    out = simulate_mve_sr(
        μ2, Σ2,
        n_obs, max_card,
        compute_sparse_mve_ES;
        max_comb = 0,
        do_checks = true
    )

    @test out === expected

    # and twice is reproducible
    Random.seed!(1234)
    out2 = simulate_mve_sr(μ2, Σ2, n_obs, max_card, compute_sparse_mve_ES;
                           max_comb = 0,
                           do_checks = true)
    @test out2 === expected
end

@testset "simulate_mve_sr error handling" begin
    μ2 = [1.0, 2.0]
    Σ2 = Matrix{Float64}(I, 2, 2)
    
    # invalid number of observations
    @test_throws AssertionError simulate_mve_sr(μ2, Σ2, 0, 1, compute_sparse_mve_ES)

    # non‐square covariance
    @test_throws AssertionError simulate_mve_sr(μ2, rand(2,3), 10, 1, compute_sparse_mve_ES)

    # invalid cardinality
    @test_throws AssertionError simulate_mve_sr(μ2, Σ2, 10, 0, compute_sparse_mve_ES)
    @test_throws AssertionError simulate_mve_sr(μ2, Σ2, 10, 3, compute_sparse_mve_ES)

    # negative max_comb
    @test_throws AssertionError simulate_mve_sr(μ2, Σ2, 10, 1, compute_sparse_mve_ES; max_comb = -5)
end
