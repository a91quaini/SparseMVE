using Test
using SparseMVE
import SparseMVE.BCW: mosek_raw_bigM, mosek_raw_MISOCP, mosek_MISOCP_relaxation

# A minimal dummy portfolio for signature checks
function make_dummy_portfolio()
    n, m, f, k = 3, 2, 2, 1
    μ  = ones(n)
    Y  = zeros(f, n)
    d  = zeros(n)
    X  = zeros(f, n)
    A  = ones(m, n)
    l  = zeros(m)
    u  = ones(m)
    γ  = ones(n)
    min_inv = zeros(n)
    return SparseMVE.BCW.SparsePortfolioData(
      μ, Y, d, X, A, l, u, k, n, m, f, min_inv, γ
    )
end

const P = make_dummy_portfolio()

@testset "raw.mosek_raw_bigM" begin
    @test isdefined(SparseMVE.BCW, :mosek_raw_bigM)
    fn = mosek_raw_bigM
    @test typeof(fn) <: Function

    # signature: expect at least (SPData, Float64, Float64, Int)
    sigs = methods(fn).ms
    found = any(m -> begin
        ps = m.sig.parameters
        length(ps) ≥ 5 &&
        ps[2] === typeof(P) &&
        ps[3] === Float64 &&
        ps[4] === Float64 &&
        ps[5] === Int
    end, sigs)
    @test found
end

@testset "raw.mosek_raw_MISOCP" begin
    @test isdefined(SparseMVE.BCW, :mosek_raw_MISOCP)
    fn = mosek_raw_MISOCP
    @test typeof(fn) <: Function

    sigs = methods(fn).ms
    found = any(m -> begin
        ps = m.sig.parameters
        length(ps) ≥ 5 &&
        ps[2] === typeof(P) &&
        ps[3] === Float64 &&
        ps[4] === Float64 &&
        ps[5] === Int
    end, sigs)
    @test found
end

@testset "raw.mosek_MISOCP_relaxation" begin
    @test isdefined(SparseMVE.BCW, :mosek_MISOCP_relaxation)
    fn = mosek_MISOCP_relaxation
    @test typeof(fn) <: Function

    sigs = methods(fn).ms
    found = any(m -> begin
        ps = m.sig.parameters
        length(ps) ≥ 3 &&
        ps[2] === typeof(P) &&
        ps[3] === Float64
    end, sigs)
    @test found
end
