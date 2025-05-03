# test/test-socp_relaxation.jl
using Test
using SparseMVE

@testset "BCW.socp_relaxation export & signature" begin
    @test isdefined(SparseMVE.BCW, :portfolios_socp)
    @test isdefined(SparseMVE.BCW, :portfolios_socp2)

    # signature check: first positional arg must be SparsePortfolioData
    sp = SparseMVE.BCW.portfolios_socp
    @test typeof(sp) <: Function
    ms = methods(sp)
    found = any(m -> begin
        ps = m.sig.parameters
        length(ps) >= 2 &&
        ps[2] === SparseMVE.BCW.SparsePortfolioData
    end, ms)
    @test found

    sp2 = SparseMVE.BCW.portfolios_socp2
    @test typeof(sp2) <: Function
    ms2 = methods(sp2)
    found2 = any(m -> begin
        ps = m.sig.parameters
        length(ps) >= 2 &&
        ps[2] === SparseMVE.BCW.SparsePortfolioData
    end, ms2)
    @test found2
end
