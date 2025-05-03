using Test
using LinearAlgebra
using SparseMVE.ES: compute_mve_sr

@testset "compute_mve_sr signature" begin
    @test isdefined(SparseMVE.ES, :compute_mve_sr)
    fn = compute_mve_sr
    @test typeof(fn) <: Function
end

@testset "compute_mve_sr functionality (full)" begin
    # Σ = [1 0; 0 4], μ = [1,2]
    # solve Σ x = μ ⇒ x = [1, 0.5], so SR = sqrt(μ⋅x) = sqrt(1*1 + 2*0.5) = sqrt(2)
    μ = [1.0, 2.0]
    Σ = [1.0 0.0; 0.0 4.0]
    sr = compute_mve_sr(μ, Σ)
    @test isapprox(sr, sqrt(2.0); atol=1e-8)
end

@testset "compute_mve_sr functionality (subset)" begin
    μ = [1.0, 2.0, 3.0]
    Σ = I(3)
    # full‐vector SR = norm(μ)
    @test isapprox(compute_mve_sr(μ, Σ), norm(μ); atol=1e-8)
    # subset indices 2 and 3
    sel = [2, 3]
    @test isapprox(compute_mve_sr(μ, Σ; selection=sel), norm(μ[sel]); atol=1e-8)
end

@testset "compute_mve_sr error handling" begin
    # mismatch mu vs Σ dimensions
    μ2 = [1.0, 2.0]
    Σ3 = I(3)
    @test_throws AssertionError compute_mve_sr(μ2, Σ3)

    # non‐square Σ
    Σ_rect = [1.0 2.0 3.0]
    @test_throws AssertionError compute_mve_sr(μ2, Σ_rect)

    # selection out of bounds
    Σ2 = I(2)
    @test_throws AssertionError compute_mve_sr(μ2, Σ2; selection=[3])
end
