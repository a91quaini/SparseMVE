using Test
using LinearAlgebra
using SparseMVE.UTILS: compute_mve_weights

@testset "compute_mve_weights signature" begin
    @test isdefined(SparseMVE.UTILS, :compute_mve_weights)
    @test isa(compute_mve_weights, Function)
end

@testset "compute_mve_weights full universe" begin
    μ = [1.0, 2.0, 3.0]
    Σ = Matrix{Float64}(I, 3, 3)
    w = compute_mve_weights(μ, Σ)
    @test w ≈ μ atol=1e-8
end

@testset "compute_mve_weights subset" begin
    μ = [1.0, 2.0, 3.0]
    Σ = Diagonal([1.0, 4.0, 9.0])
    sel = [1, 3]
    # Σ_S = diag([1,9]), μ_S = [1,3] ⇒ Σ_S⁻¹ μ_S = [1, 3/9] = [1, 1/3]
    w = compute_mve_weights(μ, Σ; selection=sel)
    @test w ≈ [1.0, 0.0, 1/3] atol=1e-8
end

@testset "compute_mve_weights error handling" begin
    μ = [1.0, 2.0]
    Σ = Matrix{Float64}(I, 2, 2)
    # dimension mismatches
    @test_throws AssertionError compute_mve_weights([1.0], Σ, do_checks=true)
    @test_throws AssertionError compute_mve_weights(μ, rand(2,3), do_checks=true)
    # invalid selection indices (must be 1 ≤ i ≤ n)
    @test_throws AssertionError compute_mve_weights(μ, Σ; selection=[0], do_checks=true)
    @test_throws AssertionError compute_mve_weights(μ, Σ; selection=[3], do_checks=true)
end
