using Test

# Directories to search for test files
test_dirs = [
    @__DIR__,
    joinpath(@__DIR__, "test-BCW"),
    joinpath(@__DIR__, "test-ES")
]

# Include all test files of the form test-[function_name].jl
for dir in test_dirs
    for (root, _, files) in walkdir(dir)
        for f in files
            if occursin(r"^test-.*\.jl$", f)
                include(joinpath(root, f))
            end
        end
    end
end