# test/test-cutting_planes_portfolios_functions.jl
using Test
using SparseMVE
using MathOptInterface
const MOI = MathOptInterface

# A tiny “dummy” portfolio with trivial data so each function runs without error.
# (You’ll want more realistic fixtures later.)
function make_dummy_portfolio()
    # n = # of assets, m = # of linear constraints, f = # of “factors” (size of α & Y)
    n, m, f, k = 3, 2, 2, 1
    μ = ones(n)             # asset expected returns
    Y = zeros(f, 1)         # factor-mean vector (length f)
    d = zeros(n)            # intercepts / RHS shifts
    X = zeros(f, n)         # factor loadings (f × n)
    A = ones(m, n)          # linear constraint matrix (m × n)
    l = zeros(m)            # constraint lower bounds
    u = ones(m)             # constraint upper bounds
    γ = ones(n)             # risk-aversion / weight
    min_inv = zeros(n)      # minimum investment per asset

    return SparseMVE.BCW.SparsePortfolioData(
      μ, Y, d, X, A, l, u,
      k,   # cardinality
      n,   # n assets
      m,   # m constraints
      f,   # f factors
      min_inv,
      γ
    )
end


const P = make_dummy_portfolio()
const idx = [1]  # single‐asset support

@testset "portfolios_objective" begin
    s = zeros(P.n); s[idx] .= 1.0
    cut = SparseMVE.BCW.portfolios_objective(P, s)
    @test isa(cut, SparseMVE.BCW.Cut)
    @test cut.status in (MOI.OPTIMAL, MOI.INFEASIBLE)
    @test length(cut.∇s) == P.n
end

@testset "portfolios_objective2" begin
    s0 = fill(0.5, P.n)
    cut2 = SparseMVE.BCW.portfolios_objective2(P, s0)
    @test isa(cut2, SparseMVE.BCW.Cut)
    @test length(cut2.∇s) == P.n
end

@testset "getWarmStart" begin
    ws = SparseMVE.BCW.getWarmStart(P, 2)
    @test isa(ws, Vector{Float64})
    @test sum(ws) <= P.k  # should respect cardinality
    @test length(ws) == P.n
end

@testset "cutting_planes_portfolios signature" begin
    fn = SparseMVE.BCW.cutting_planes_portfolios
    @test typeof(fn) <: Function

    # Check there's a method taking (SparsePortfolioData, Float64, Float64)
    mlist = methods(fn)
    found = any(m -> begin
        ps = m.sig.parameters
        length(ps) >= 4 &&
          ps[2] === SparseMVE.BCW.SparsePortfolioData &&
          ps[3] === Float64 &&
          ps[4] === Float64
    end, mlist)
    @test found
end

