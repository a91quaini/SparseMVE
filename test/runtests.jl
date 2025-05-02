# Run *all* .jl files in this directory except this one:
for file in filter(f->endswith(f, ".jl") && f != basename(@__FILE__), readdir(@__DIR__))
    include(joinpath(@__DIR__, file))
end
