using Test
using LinearAlgebra
using SparseMVE.ES: compute_mve_weights

@testset "compute_mve_weights signature" begin
    @test isdefined(SparseMVE.ES, :compute_mve_weights)
    fn = compute_mve_weights
    @test typeof(fn) <: Function
end

@testset "compute_mve_weights full universe" begin
    μ = [1.0, 2.0, 3.0]
    Σ = Matrix{Float64}(I, 3, 3)
    w = compute_mve_weights(μ, Σ)
    @test w ≈ μ atol=1e-8
end

@testset "compute_mve_weights with gamma" begin
    μ = [1.0, 2.0]
    Σ = [2.0 0.0; 0.0 4.0]
    # Σ⁻¹ * μ = [1/2, 2/4] = [0.5, 0.5], gamma=2 ⇒ [0.25, 0.25]
    w = compute_mve_weights(μ, Σ; gamma=2.0)
    @test w ≈ [0.25, 0.25] atol=1e-8
end

@testset "compute_mve_weights subset" begin
    μ = [1.0, 2.0, 3.0]
    Σ = Diagonal([1.0, 4.0, 9.0])
    sel = [1, 3]
    # Σ_S = diag([1,9]), μ_S = [1,3], solve Σ_S⁻¹ μ_S = [1, 3/9] = [1, 1/3]
    w = compute_mve_weights(μ, Σ; selection=sel)
    @test w ≈ [1.0, 0.0, 1/3] atol=1e-8
end

@testset "compute_mve_weights error handling" begin
    μ = [1.0, 2.0]
    Σ = Matrix{Float64}(I, 2, 2)
    # mismatched dimensions
    @test_throws AssertionError compute_mve_weights([1.0], Σ)
    @test_throws AssertionError compute_mve_weights(μ, rand(2,3))
    # non-positive gamma
    @test_throws AssertionError compute_mve_weights(μ, Σ; gamma=0.0)
    # invalid selections (1-based indexing)
    @test_throws AssertionError compute_mve_weights(μ, Σ; selection=[0])
    @test_throws AssertionError compute_mve_weights(μ, Σ; selection=[3])
end
