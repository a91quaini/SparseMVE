using Test
using SparseMVE    # brings in BCW
using LinearAlgebra  # for any utility if needed

@testset "BCW.hillclimb export & signature" begin

    # 1. It’s exported
    @test isdefined(SparseMVE.BCW, :portfolios_hillclimb)
    @test typeof(SparseMVE.BCW.portfolios_hillclimb) <: Function

    # 2. There is a method whose 2nd arg is SparsePortfolioData and 3rd is Vector{Int}
    found = any(m -> begin
        params = m.sig.parameters
        length(params) >= 3 &&
          params[2] === SparseMVE.BCW.SparsePortfolioData &&
          params[3] === Vector{Int}
    end, methods(SparseMVE.BCW.portfolios_hillclimb))

    @test found
end
