module UTILS

using LinearAlgebra

include(joinpath(@__DIR__,"UTILS","compute_sr.jl"))
include(joinpath(@__DIR__,"UTILS","compute_mve_sr.jl"))
include(joinpath(@__DIR__,"UTILS","compute_mve_weights.jl"))
include(joinpath(@__DIR__,"UTILS","compute_mve_sr_decomposition.jl"))

# export your function
export compute_sr, 
       compute_mve_sr,
       compute_mve_weights,
       compute_mve_sr_decomposition

end # module UTILS