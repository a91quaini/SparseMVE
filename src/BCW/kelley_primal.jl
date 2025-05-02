using JuMP
using MosekTools
using LinearAlgebra

function getKelleyPrimalCuts(
    thePortfolio::SparsePortfolioData,
    useCopyOfVariables::Bool,
    minReturnConstraint::Bool,
    theStabilizationPoint::Vector{Float64},
    kelleyPrimalEpochs::Int)

    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)

    @variable(model, sRoot[1:thePortfolio.n] >= 0)
    @constraint(model, sRoot .<= 1)
    @variable(model, tRoot >= -1e12)

    @objective(model, Min, tRoot)

    @constraint(model, sum(sRoot) <= thePortfolio.k)
    @constraint(model, sum(sRoot) >= 1)

    if useCopyOfVariables
        @constraint(model,
            sum(sRoot[i] * (thePortfolio.A[1,i] > thePortfolio.l[1]) for i in 1:thePortfolio.n) >= 1.0)
        @variable(model, xRoot[1:thePortfolio.n] >= 0)
        @constraint(model, thePortfolio.A * xRoot .>= thePortfolio.l)
        @constraint(model, thePortfolio.A * xRoot .<= thePortfolio.u)
        @constraint(model, [i=1:thePortfolio.n], xRoot[i] >= sRoot[i] * thePortfolio.min_investment[i])
        @constraint(model, [i=1:thePortfolio.n], xRoot[i] <= thePortfolio.u[i+1] * sRoot[i])
        @constraint(model, sum(xRoot) == 1.0)
    end

    if minReturnConstraint
        @constraint(model,
            sum(sRoot[i] * (thePortfolio.A[1,i] > thePortfolio.l[1]) for i in 1:thePortfolio.n) >= 1.0)
    end

    UB = Inf
    LB = -Inf
    rootStabilizationTrick = :inOut
    ε = 1e-10
    λ = (rootStabilizationTrick == :inOut) ? 0.1 : 1.0
    δ = (rootStabilizationTrick in (:inOut, :twoEps)) ? 2*ε : 0.0
    rootCutsLim = kelleyPrimalEpochs

    theCutPool = Cut[]
    stabilizationPoint = copy(theStabilizationPoint)
    consecutiveNonImprov_1 = 0
    consecutiveNonImprov_2 = 0

    for epoch in 1:kelleyPrimalEpochs
        optimize!(model)
        zstar = clamp.(value.(sRoot), 0.0, 1.0)
        stabilizationPoint .= (stabilizationPoint .+ zstar) ./ 2

        currentObj = objective_value(model)
        if LB >= currentObj - eps()
            if consecutiveNonImprov_1 == 5
                consecutiveNonImprov_2 += 1
            else
                consecutiveNonImprov_1 += 1
            end
        else
            consecutiveNonImprov_1 = 0
            consecutiveNonImprov_2 = 0
        end
        LB = max(LB, currentObj)

        if consecutiveNonImprov_1 == 5
            λ = 1.0
        elseif consecutiveNonImprov_2 == 5
            δ = 0.0
        end

        z0 = clamp.(λ .* zstar .+ (1-λ) .* stabilizationPoint .+ δ, 0.0, 1.0)

        cut = portfolios_objective2(thePortfolio, z0)
        addOACut = (cut.status == :Optimal)

        if cut.status == :Optimal
            UB = min(UB, cut.p)
        else
            cut = portfolios_objective2(thePortfolio, z0)
        end

        if cut.status == :Optimal
            @constraint(model, tRoot >= cut.p + dot(cut.∇s, sRoot .- z0))
            if epoch <= rootCutsLim && addOACut
                cut.p += dot(cut.∇s, -z0)
                push!(theCutPool, cut)
            end
        else
            @constraint(model,
                sum(z0[i]*(1.0-sRoot[i]) + sRoot[i]*(1.0-z0[i]) for i in 1:thePortfolio.n) >= 1.0)
            stabilizationPoint .= theStabilizationPoint
        end
    end

    return theCutPool
end
