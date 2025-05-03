# test/test-inner_problem.jl
using Test
using SparseMVE

@testset "BCW.inner_problem export & signature" begin
    @test isdefined(SparseMVE.BCW, :inner_dual)
    @test isdefined(SparseMVE.BCW, :inner_dual2)

    methods_list = methods(SparseMVE.BCW.inner_dual)
    found_dual = any(m -> begin
        params = m.sig.parameters
        length(params) >= 3 &&
          params[2] === SparseMVE.BCW.SparsePortfolioData &&
          params[3] === Vector{Int}
    end, methods_list)
    @test found_dual

    methods_list2 = methods(SparseMVE.BCW.inner_dual2)
    found_dual2 = any(m -> begin
        params = m.sig.parameters
        length(params) >= 4 &&
          params[2] === SparseMVE.BCW.SparsePortfolioData &&
          params[3] === Vector{Int} &&
          params[4] === Vector{Float64}
    end, methods_list2)
    @test found_dual2
end
