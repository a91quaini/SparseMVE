module ES

using LinearAlgebra

include(joinpath(@__DIR__,"ES","compute_sr.jl"))
include(joinpath(@__DIR__,"ES","compute_mve_sr.jl"))
include(joinpath(@__DIR__,"ES","compute_mve_weights.jl"))
include(joinpath(@__DIR__,"ES","compute_mve_sr_cardk.jl"))

# export your function
export compute_sr, 
       compute_mve_sr,
       compute_mve_weights,
       compute_mve_sr_cardk

end # module ES