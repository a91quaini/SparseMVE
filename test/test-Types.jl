using Test
using SparseMVE        # pulls in BCW.Types
using LinearAlgebra    # for I, if you need it

@testset "BCW.Types" begin

    # -- CutIterData --
    ci = SparseMVE.CutIterData(0.1, 2,    # time, cut_count
    3.0, 4.0, # obj, best_bound
    1.5,      # bound_gap
    :ok)      # status

    @test isa(ci, SparseMVE.CutIterData)
    @test ci.time       == 0.1
    @test ci.cut_count  == 2
    @test ci.status     == :ok

    # -- Cut (mutable) --
    c = SparseMVE.Cut(0.2, [1.0, 2.0], :active)
    @test isa(c, SparseMVE.Cut)
    @test c.p           == 0.2
    @test c.grad_s      == [1.0, 2.0]

    # -- Dual --
    α = [0.1, 0.2]; βl = [0.0]; βu = [1.0]; ρ = [0.3]; w = [0.4]
    d = SparseMVE.Dual(α, 1.5, βl, βu, ρ, w, 2.5, :dual)
    @test isa(d, SparseMVE.Dual)
    @test d.ofv         == 2.5
    @test d.α           == α

    # -- SparsePortfolioData --
    μ  = [0.1]
    Y  = [1.0 2.0; 3.0 4.0]
    d0 = [0.2]
    X  = Y
    A  = Matrix{Float64}(I,2,2)
    l  = [0.0]; u = [1.0]
    spd = SparseMVE.SparsePortfolioData(μ, Y, d0, X, A, l, u,
                1, 1, 1, 1,
                [0.1], [0.2])
    @test isa(spd, SparseMVE.SparsePortfolioData)
    @test spd.n         == 1
    @test spd.γ         == [0.2]

    # -- CardinalityConstrainedPortfolio --
    ccp = SparseMVE.CardinalityConstrainedPortfolio(
    [1], [0.5],    # indices, w
    1.0,           # λ
    [0.1], [0.0], [0.0], [0.2],  # α, βl, βu, ρ
    0.5, 0.6,      # bound_gap, bound_gap_socp
    1,             # num_iters
    0.05, 0.01,    # expected_return, portfolio_variance
    0.1, 1         # solve_time, node_count
    )
    @test isa(ccp, SparseMVE.CardinalityConstrainedPortfolio)
    @test ccp.solve_time == 0.1
    @test ccp.node_count == 1

    # -- PortfolioData --
    pd = SparseMVE.PortfolioData(
    [0.1],
    [1.0 2.0;3.0 4.0],
    [1.0 0.0;0.0 1.0],
    [1.0 2.0;3.0 4.0],
    [0.1]
    )
    @test isa(pd, SparseMVE.PortfolioData)
    @test length(pd.min_investment) == 1

end