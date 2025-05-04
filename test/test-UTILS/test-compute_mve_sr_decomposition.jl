using Test
using LinearAlgebra

# pull in both the decomposition routine and one sparse‐MVE implementation
using SparseMVE.UTILS: compute_mve_sr_decomposition
using SparseMVE.ES: compute_sparse_mve_ES

# a tiny helper that just forwards to compute_sparse_mve_ES
const sparse_mve = compute_sparse_mve_ES

# common 2‐asset identity test data
const μ_pop     = [1.0, 2.0]
const Σ_pop     = I(2)
const μ_sample  = [3.0, 4.0]
const Σ_sample  = I(2)

@testset "compute_mve_sr_decomposition signature" begin
    @test isdefined(SparseMVE.UTILS, :compute_mve_sr_decomposition)
    fn = compute_mve_sr_decomposition
    @test typeof(fn) <: Function
end

@testset "compute_mve_sr_decomposition functionality (k=1)" begin
    out = compute_mve_sr_decomposition(
        μ_pop, Σ_pop,
        μ_sample, Σ_sample,
        1, sparse_mve; max_comb=0, do_checks=true
    )
    @test isa(out, NamedTuple)
    @test isa(out, NamedTuple)
    expected_keys = [
        :sample_mve_sr,
        :sample_sparse_mve_sr,
        :sparse_mve_sr_est_term,
        :sparse_mve_sr_sel_term,
    ]
    @test sort(collect(keys(out))) == sort(expected_keys)

    # 1) unconstrained sample‐MVE SR = sqrt(3²+4²) = 5
    @test isapprox(out.sample_mve_sr, 5.0; atol=1e-8)

    # 2) best‐1‐asset sample‐MVE SR = max(3,4) = 4
    @test isapprox(out.sample_sparse_mve_sr, 4.0; atol=1e-8)

    # 3) est_term = apply w_sample=[0,4] to pop ⇒ Sharpe = pop μ⋅w / √(w'Σw) = 2/1 = 2
    @test isapprox(out.sparse_mve_sr_est_term, 2.0; atol=1e-8)

    # 4) sel_term = best‐1‐asset on pop = max(pop μ) = 2
    @test isapprox(out.sparse_mve_sr_sel_term, 2.0; atol=1e-8)
end

@testset "compute_mve_sr_decomposition functionality (k=2)" begin
    out = compute_mve_sr_decomposition(
        μ_pop, Σ_pop,
        μ_sample, Σ_sample,
        2, sparse_mve;
        max_comb = 0,
        do_checks = true
    )

    # 1) unconstrained sample‐MVE SR = 5
    @test isapprox(out.sample_mve_sr, 5.0; atol=1e-8)
    # 2) best‐2‐asset sample‐MVE SR = sample full‐support = 5
    @test isapprox(out.sample_sparse_mve_sr, 5.0; atol=1e-8)
    # 3) est term: apply w_sample=[3,4] to pop ⇒ pop‐Sharpe = (1*3+2*4)/√(3²+4²)=11/5=2.2
    @test isapprox(out.sparse_mve_sr_est_term, 11/5; atol=1e-8)
    # 4) sel term: best‐2‐asset on pop = √(1²+2²)=√5
    @test isapprox(out.sparse_mve_sr_sel_term, sqrt(5); atol=1e-8)
end

@testset "compute_mve_sr_decomposition error handling" begin
    # dimension mismatches
    @test_throws AssertionError compute_mve_sr_decomposition(
        [1.0], I(2),
        μ_sample, Σ_sample,
        1, sparse_mve
    )

    @test_throws AssertionError compute_mve_sr_decomposition(
        μ_pop, Σ_pop,
        [1.0,2.0,3.0], I(2),
        1, sparse_mve
    )

    # invalid cardinality
    @test_throws AssertionError compute_mve_sr_decomposition(
        μ_pop, Σ_pop,
        μ_sample, Σ_sample,
        0, sparse_mve
    )

    @test_throws AssertionError compute_mve_sr_decomposition(
        μ_pop, Σ_pop,
        μ_sample, Σ_sample,
        3, sparse_mve
    )

    # negative max_comb
    @test_throws AssertionError compute_mve_sr_decomposition(
        μ_pop, Σ_pop,
        μ_sample, Σ_sample,
        1, sparse_mve;
        max_comb = -1
    )
end
