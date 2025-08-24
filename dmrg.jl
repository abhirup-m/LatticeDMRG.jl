using LinearAlgebra, ProgressMeter


function InitOperators()
    Sp = [0.0 1.0;
          0.0 0.0]
    Sm = [0.0 0.0;
          1.0 0.0]
    Sz = [1.0 0.0;
          0.0 -1.0]
    return Dict('z' => Sz, '+' => Sp, '-' => Sm)
end


function InfiniteDMRG(
        numSteps::Int64,
        bondDim::Int64,
    )
    σ = InitOperators()

    sysId = I(2)
    envId = I(2)

    sysHam = 0 .* sysId
    envHam = 0 .* envId

    sysOps = copy(σ)
    envOps = copy(σ)

    results = Dict("vals" => Float64[], "vecs" => Vector{Float64}[])

    @showprogress desc="bond dimension=$(bondDim)" for step in 1:numSteps
        sysExpId = kron(sysId, I(2))
        envExpId = kron(I(2), envId)

        sysExpHam = kron(sysHam, I(2)) + 0.25 * kron(sysOps['z'], σ['z']) + 0.5 * kron(sysOps['+'], σ['-']) + 0.5 * kron(sysOps['-'], σ['+'])
        envExpHam = kron(I(2), envHam) + 0.25 * kron(σ['z'], envOps['z']) + 0.5 * kron(σ['+'], envOps['-']) + 0.5 * kron(σ['-'], envOps['+'])

        sysExpOps = Dict(k => kron(sysId, op) for (k,op) in σ)
        envExpOps = Dict(k => kron(op, envId) for (k,op) in σ)

        superHam = kron(sysExpId, envExpHam) + kron(sysExpHam, envExpId)
        superHam += 0.25 * kron(sysExpOps['z'], envExpOps['z']) + 0.5 * kron(sysExpOps['+'], envExpOps['-']) + 0.5 * kron(sysExpOps['-'], envExpOps['+']) 

        vals, vecs = eigen(Hermitian(superHam))
        push!(results["vals"], vals[1])
        push!(results["vecs"], vecs[:, 1])

        groundState = vecs[:, 1]
        groundStateTensor = reshape(groundState, (size(envExpHam)[1], size(sysExpHam)[1]))'
        F = svd(groundStateTensor)
        sysRotate = F.U[:, 1:minimum((bondDim, length(F.S)))]
        envRotate = F.V[:, 1:minimum((bondDim, length(F.S)))]

        sysId = sysRotate' * sysExpId * sysRotate
        envId = envRotate' * envExpId * envRotate
        sysHam = sysRotate' * sysExpHam * sysRotate
        envHam = envRotate' * envExpHam * envRotate
        sysOps = Dict(k => sysRotate' * op * sysRotate for (k, op) in sysExpOps)
        envOps = Dict(k => envRotate' * op * envRotate for (k, op) in envExpOps)
    end
    return results

end
