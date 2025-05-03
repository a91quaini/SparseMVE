module ES

using LinearAlgebra

include(joinpath(@__DIR__,"ES","compute_sr.jl"))
include(joinpath(@__DIR__,"ES","compute_mve_sr.jl"))

# export your function
export compute_sr, 
       compute_mve_sr

end # module ES