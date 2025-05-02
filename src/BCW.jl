module BCW

# bring in any dependencies your sub‐modules expect
using JuMP
using MosekTools
using LinearAlgebra

# pull in your type and algorithm files
include(joinpath(@__DIR__, "BCW", "Types.jl"))
include(joinpath(@__DIR__, "BCW", "hillclimb.jl"))

# re‐export everything
export CutIterData,
       Cut,
       Dual,
       SparsePortfolioData,
       CardinalityConstrainedPortfolio,
       PortfolioData,
       portfolios_hillclimb

end # module BCW