using Random
using SparseMVE.BCW: mve_weights_l0_BCW

#–– Parameters ––
T = 100   # number of observations
n = 10    # number of assets
k = 3     # cardinality
seed = 2025

#–– Generate synthetic returns ––
Random.seed!(seed)
returns = randn(T, n)

#–– Solve sparse MV ––
w = mve_weights_l0_BCW(returns, k)

#–– Display ––
println("mve_weights_l0_BCW with T=$T, n=$n, k=$k, seed=$seed")
println("  Nonzeros: ", count(x->abs(x)>1e-8, w))
println("  Weights:   ", w)
