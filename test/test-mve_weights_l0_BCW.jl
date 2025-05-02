using Test, Random
using SparseMVE.BCW: mve_weights_l0_BCW

@testset "mve_weights_l0_BCW signature" begin
    @test isdefined(SparseMVE.BCW, :mve_weights_l0_BCW)
    fn = mve_weights_l0_BCW
    @test typeof(fn) <: Function

    methods_list = methods(fn)
    found = any(m -> begin
        ps = m.sig.parameters
        length(ps) >= 2 && ps[2] <: AbstractMatrix
    end, methods_list)
    @test found
end

@testset "mve_weights_l0_BCW functionality" begin
    Random.seed!(1234)
    # Generate a small random returns matrix (20 obs × 4 assets)
    T, n = 20, 4
    returns = randn(T, n)

    # Test default cardinality = 1
    w1 = mve_weights_l0_BCW(returns)
    @test length(w1) == n
    @test count(x -> abs(x) > 1e-8, w1) <= 1      # at most one nonzero
    @test abs(sum(w1) - 1.0) < 1e-6               # sums to one
    @test all(w1 .>= -1e-8)                       # nonnegative up to tiny tol

    # Test cardinality = 2
    w2 = mve_weights_l0_BCW(returns, 2)
    @test length(w2) == n
    @test count(x -> abs(x) > 1e-8, w2) <= 2      # at most two nonzeros
    @test abs(sum(w2) - 1.0) < 1e-6               # sums to one
    @test all(w2 .>= -1e-8)                       # nonnegative
end