# test/test-kelley_primal.jl
using Test
using SparseMVE

@testset "BCW.getKelleyPrimalCuts export & signature" begin
    @test isdefined(SparseMVE.BCW, :getKelleyPrimalCuts)
    @test typeof(SparseMVE.BCW.getKelleyPrimalCuts) <: Function

    found = any(m -> begin
        params = m.sig.parameters
        length(params) >= 6 &&
        params[2] === SparseMVE.BCW.SparsePortfolioData &&
        params[3] === Bool &&
        params[4] === Bool &&
        params[5] === Vector{Float64} &&
        params[6] === Int
    end, methods(SparseMVE.BCW.getKelleyPrimalCuts))

    @test found
end
