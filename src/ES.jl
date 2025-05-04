module ES 
# Module implementing exhaustive search

using LinearAlgebra
using SparseMVE.UTILS: compute_sr, 
                       compute_mve_sr,
                       compute_mve_weights

include(joinpath(@__DIR__,"ES","compute_sparse_mve_ES.jl"))

# export your function
export compute_sparse_mve_ES

end # module ES