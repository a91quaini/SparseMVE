using Test
using LinearAlgebra
using SparseMVE.UTILS: compute_sr

@testset "compute_sr export & signature" begin
    @test isdefined(SparseMVE.ES, :compute_sr)
    fn = compute_sr
    @test typeof(fn) <: Function
end

@testset "compute_sr functionality" begin
    # Simple 2-asset example with identity covariance
    w1 = [1.0, 0.0]
    mu = [3.0, 4.0]
    sigma = Matrix{Float64}(I, 2, 2)
    @test compute_sr(w1, mu, sigma) ≈ 3.0

    # Equal weights
    w2 = [0.5, 0.5]
    expected = (0.5*3.0 + 0.5*4.0) / sqrt(0.5^2 + 0.5^2)
    @test compute_sr(w2, mu, sigma) ≈ expected

    # Subset selection: only asset 2
    @test compute_sr([0.3, 0.7], mu, sigma, selection=[2]) ≈ 4.0
end

@testset "compute_sr error handling" begin
        # Empty inputs
    @test_throws MethodError compute_sr([], [], zeros(0,0))

    # Length mismatch between w and mu between w and mu
    @test_throws AssertionError compute_sr([1.0], [1.0], zeros(2,2))

    # Covariance dimension mismatch
    @test_throws AssertionError compute_sr([1.0,2.0], [1.0,2.0], zeros(1,2))

        # Covariance dimension mismatch
    @test_throws AssertionError compute_sr([1.0,2.0], [1.0,2.0], zeros(1,2))

        # Selection out of bounds
    sigma2 = Matrix{Float64}(I, 2, 2)
    @test_throws AssertionError compute_sr([1.0,2.0], [1.0,2.0], sigma2, selection=[3])
end