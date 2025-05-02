using LinearAlgebra, Statistics

"""
    mve_weights_l0_BCW(returns::AbstractMatrix, card_k::Integer=1)

Compute the cardinality-constrained mean-variance portfolio weights with at most
`card_k` nonzeros via the cutting-plane method of Bertsimas & Cory-Wright (2022).
- `returns`: T×n matrix of asset returns.
- `card_k`: number of nonzero weights (default 1).
"""
function mve_weights_l0_BCW(
    returns::AbstractMatrix{<:Real},
    card_k::Integer = 1
)
    # 1) sample mean and covariance
    μ = vec(mean(returns; dims=1))                # length n
    Σ = cov(returns; dims=1)                      # n×n

    # 2) ensure PD, compute Cholesky Σ = X'X
    n = length(μ)
    Σpd = Σ + 1e-8 * I(n)
    C = cholesky(Σpd).U                         # upper-triangular n×n
    X = C'                                      # n×n

    # 3) compute regression data y, d
    proj = (X * X') \ (X * μ)                   # length n vector
    y_vec = X * proj                             # length n vector
    Y = reshape(y_vec, n, 1)                     # n×1 matrix
    d = (X' * proj) .- μ                         # length n vector

    # 4) no extra linear constraints
    A = zeros(Float64, 0, n)
    l = zeros(Float64, 0)
    u = zeros(Float64, 0)
    min_inv = zeros(n)
    γ = ones(n)
    m = 0
    f = size(X, 1)

    # 5) build portfolio data and solve
    P = SparsePortfolioData(
        μ, Y, d, X, A, l, u,
        card_k,
        n,
        m,
        f,
        min_inv,
        γ
    )

    sol = cutting_planes_portfolios(P)

    # build full‐length weight vector
    n = size(returns, 2)               # #assets
    w_full = zeros(n)
    w_full[sol.indices] .= sol.w
    return w_full
end

