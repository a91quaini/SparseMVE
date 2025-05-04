using Test
using LinearAlgebra
using SparseMVE.ES: compute_sparse_mve_ES

# simple 2‐asset identity‐covariance example
const μ2 = [3.0, 4.0]
const Σ2 = I(2)

@testset "compute_sparse_mve_ES signature" begin
    @test isdefined(SparseMVE.ES, :compute_sparse_mve_ES)
    @test isa(compute_sparse_mve_ES, Function)
end

@testset "compute_sparse_mve_ES functionality (k=1)" begin
    out = compute_sparse_mve_ES(μ2, Σ2, 1; max_comb=0)
    @test isa(out, NamedTuple)
    @test keys(out) == (:sr, :weights, :selection)
    # With k=1, best single‐asset Sharpe is max(μ)/√1 = 4.0
    @test out.sr ≈ 4.0
    @test out.selection == [2]
    @test length(out.weights) == 2
    @test out.weights ≈ [0.0, 4.0]
end

@testset "compute_sparse_mve_ES functionality (k=2)" begin
    out = compute_sparse_mve_ES(μ2, Σ2, 2; max_comb=0)
    # Now best full‐support Sharpe = √(3²+4²)=5
    @test out.sr ≈ 5.0 atol=1e-8
    @test sort(out.selection) == [1,2]
    @test length(out.weights) == 2
    # weights = Σ⁻¹ μ = μ
    @test out.weights ≈ [3.0, 4.0]
end

@testset "compute_sparse_mve_ES error handling" begin
    # non‐square Σ
    @test_throws AssertionError compute_sparse_mve_ES(μ2, rand(2,3), 1)
    # length mismatch μ vs Σ
    @test_throws AssertionError compute_sparse_mve_ES([1.0,2.0,3.0], I(2), 1)
    # invalid cardinality
    @test_throws AssertionError compute_sparse_mve_ES(μ2, Σ2, 0)
    @test_throws AssertionError compute_sparse_mve_ES(μ2, Σ2, 3)
    # negative max_comb
    @test_throws AssertionError compute_sparse_mve_ES(μ2, Σ2, 1; max_comb=-1)
end
