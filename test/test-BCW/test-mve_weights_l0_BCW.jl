using Test
using Random
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
    Random.seed!(42)
    # small test: 30 observations of 5 assets
    T, n = 30, 5
    returns = randn(T, n)

    # cardinality = 1 (default)
    w1 = mve_weights_l0_BCW(returns)
    @test length(w1) == n
    @test count(x -> abs(x) > 1e-6, w1) <= 1      # sparsity
        @test count(x -> abs(x) > 1e-6, w1) <= 1      # sparsity

    # cardinality = 2
    w2 = mve_weights_l0_BCW(returns, 2)
    @test length(w2) == n
    @test count(x -> abs(x) > 1e-6, w2) <= 2      # sparsity
end