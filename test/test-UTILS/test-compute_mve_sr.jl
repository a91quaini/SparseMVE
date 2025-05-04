using Test
using LinearAlgebra
using SparseMVE.UTILS: compute_mve_sr

@testset "compute_mve_sr signature" begin
    @test isdefined(SparseMVE.ES, :compute_mve_sr)
    @test isa(compute_mve_sr, Function)
end

@testset "compute_mve_sr functionality (full universe)" begin
    # Σ = [1 0; 0 4], μ = [1,2]
    μ = [1.0, 2.0]
    Σ = [1.0 0.0; 0.0 4.0]
    expected_sr = sqrt(2.0)  # as in Σ \ μ = [1,0.5], μ'·[1,0.5] = 2
    @test isapprox(compute_mve_sr(μ, Σ), expected_sr; atol=1e-8)
end

@testset "compute_mve_sr functionality (subset)" begin
    μ = [1.0, 2.0, 3.0]
    # build a Float64 identity matrix explicitly
    Σ = Matrix{Float64}(I, 3, 3)
    # full‐vector SR = √(μ'·(Σ⁻¹μ)) but here Σ=I so SR = norm(μ)
    @test isapprox(compute_mve_sr(μ, Σ), norm(μ); atol=1e-8)

    # now test with a subset of assets
    sel = [2, 3]
    @test isapprox(compute_mve_sr(μ, Σ; selection=sel), norm(μ[sel]); atol=1e-8)
end

@testset "compute_mve_sr error handling" begin
    # dimension mismatch between μ and Σ
    μ2 = [1.0, 2.0]
    Σ3 = Matrix{Float64}(I, 3, 3)
    @test_throws AssertionError compute_mve_sr(μ2, Σ3)

    # non‐square Σ
    Σ_rect = [1.0 2.0 3.0]
    @test_throws AssertionError compute_mve_sr(μ2, Σ_rect)

    # selection indices out of bounds
    Σ2 = Matrix{Float64}(I, 2, 2)
    @test_throws AssertionError compute_mve_sr(μ2, Σ2; selection=[3])
end
