module BCW
# Module implementing the approach by 
# Bertsimas, Dimitris and Cory-Wright, Ryan
# A scalable algorithm for sparse portfolio selection
# 2022

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
include(joinpath(@__DIR__, "BCW", "socp_relaxation.jl"))
include(joinpath(@__DIR__,"BCW","raw.jl"))
include(joinpath(@__DIR__,"BCW","mve_weights_l0_BCW.jl"))

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
       cutting_planes_portfolios,
       portfolios_socp, 
       portfolios_socp2,
       mosek_raw_bigM,
       mosek_raw_MISOCP,
       mosek_MISOCP_relaxation,
       mve_weights_l0_BCW

end # module BCW