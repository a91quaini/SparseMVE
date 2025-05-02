struct CutIterData
    time::Float64
    cut_count::Int
    obj::Float64
    best_bound::Float64
    bound_gap::Float64
    status::Symbol
end

mutable struct Cut
    p::Float64
    grad_s::Vector{Float64}
    status::Symbol
end

struct Dual
    α::Vector{Float64}
    λ::Float64
    βl::Vector{Float64}
    βu::Vector{Float64}
    ρ::Vector{Float64}
    w::Vector{Float64}
    ofv::Float64
    status::Symbol
end

struct SparsePortfolioData
    μ::Vector{Float64}
    Y::Matrix{Float64}
    d::Vector{Float64}
    X::Matrix{Float64}
    A::Matrix{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    k::Int             # cardinality
    n::Int             # number of assets
    m::Int             # number of constraints
    f::Int             # rank of covariance
    min_investment::Vector{Float64}
    γ::Vector{Float64}
end

struct CardinalityConstrainedPortfolio
    indices::Vector{Int}
    w::Vector{Float64}
    λ::Float64
    α::Vector{Float64}
    βl::Vector{Float64}
    βu::Vector{Float64}
    ρ::Vector{Float64}
    bound_gap::Float64
    bound_gap_socp::Float64
    num_iters::Int
    expected_return::Float64
    portfolio_variance::Float64
    solve_time::Float64
    node_count::Int
end

struct PortfolioData
    μ::Vector{Float64}
    X::Matrix{Float64}
    normalization::Matrix{Float64}
    X_normalized::Matrix{Float64}
    min_investment::Vector{Float64}
end
