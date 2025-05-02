module BCW

# bring in any dependencies your sub‐modules expect
using JuMP
using MosekTools
using LinearAlgebra

# pull in your type and algorithm files
include(joinpath(@__DIR__, "BCW", "Types.jl"))
include(joinpath(@__DIR__, "BCW", "hillclimb.jl"))
include(joinpath(@__DIR__, "BCW", "inner_problem.jl"))
include(joinpath(@__DIR__,"BCW","kelley_primal.jl"))
include(joinpath(@__DIR__,"BCW","miop_formulation.jl"))

# re‐export everything
export CutIterData,
       Cut,
       Dual,
       SparsePortfolioData,
       CardinalityConstrainedPortfolio,
       PortfolioData,
       portfolios_hillclimb
       inner_dual,
       inner_dual2,
       getKelleyPrimalCuts,
       cutting_planes_portfolios

end # module BCW