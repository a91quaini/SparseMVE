using Test
using LinearAlgebra
using SparseMVE.ES: compute_mve_sr_cardk

# simple 2‐asset identity‐covariance example
const μ2 = [3.0, 4.0]
const Σ2 = I(2)

@testset "compute_mve_sr_cardk signature" begin
    @test isdefined(SparseMVE.ES, :compute_mve_sr_cardk)
    fn = compute_mve_sr_cardk
    @test typeof(fn) <: Function
end

@testset "compute_mve_sr_cardk functionality (k=1)" begin
    out = compute_mve_sr_cardk(μ2, Σ2, 1; max_comb=0)
    @test isa(out, NamedTuple)
    @test keys(out) == (:sr, :weights, :selection)
    # With k=1, best single‐asset Sharpe is max(μ)/√1 = 4.0
    @test out.sr ≈ 4.0
    @test out.selection == [2]
    @test length(out.weights) == 2
    @test out.weights ≈ [0.0, 4.0]
end

@testset "compute_mve_sr_cardk functionality (k=2)" begin
    out = compute_mve_sr_cardk(μ2, Σ2, 2; max_comb=0)
    # Now best full‐support Sharpe = √(3²+4²)=5
    @test out.sr ≈ 5.0 atol=1e-8
    @test sort(out.selection) == [1,2]
    @test length(out.weights) == 2
    # weights = Σ⁻¹ μ = μ
    @test out.weights ≈ [3.0, 4.0]
end

@testset "compute_mve_sr_cardk with ridge (gamma)" begin
    # tiny ridge won't change the argmax support, but will alter sr slightly
    out0 = compute_mve_sr_cardk(μ2, Σ2, 2; max_comb=0, gamma=0.0)
    out1 = compute_mve_sr_cardk(μ2, Σ2, 2; max_comb=0, gamma=1e-6)
    @test out1.selection == out0.selection
    @test out1.sr ≈ out0.sr atol=1e-6
end

@testset "compute_mve_sr_cardk error handling" begin
    # non‐square Σ
    @test_throws AssertionError compute_mve_sr_cardk(μ2, rand(2,3), 1)
    # length mismatch μ vs Σ
    @test_throws AssertionError compute_mve_sr_cardk([1.0,2.0,3.0], I(2), 1)
    # invalid cardinality
    @test_throws AssertionError compute_mve_sr_cardk(μ2, Σ2, 0)
    @test_throws AssertionError compute_mve_sr_cardk(μ2, Σ2, 3)
    # negative max_comb
    @test_throws AssertionError compute_mve_sr_cardk(μ2, Σ2, 1; max_comb=-1)
    # negative gamma
    @test_throws AssertionError compute_mve_sr_cardk(μ2, Σ2, 1; gamma=-0.1)
end