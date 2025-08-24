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


function ManifestMatrix(
        operator,
        siteOperators,
    )
    matrix = zeros(size(collect(values(siteOperators))[1])...)
    for (opDef, sites, coupling) in operator
        matrix .+= coupling * prod(siteOperators[(ch, site)] for (ch, site) in zip(opDef, sites))
    end
    return matrix
end


function Glue(
        operator,
        sysOps,
        envOps,
    )
    sysId = I(size(collect(values(sysOps))[1])[1])
    envId = I(size(collect(values(envOps))[1])[1])
    matrix = Matrix(0. * kron(sysId, envId))
    for (opDef, sites, coupling) in operator
        matrix .+= coupling * prod(site > 0 ? kron(sysOps[(ch, site)], envId) : kron(sysId, envOps[(ch, -site)]) for (ch, site) in zip(opDef, sites))
    end
    return matrix
end

function InfiniteDMRG(
        maxSites::Int64,
        bondDim::Int64,
        locHam::Vector,
        incHam,
        glueHam,
    )
    sysOps = Dict((k, 1) => v for (k,v) in InitOperators())
    envOps = copy(sysOps)

    sysId = I(2)
    envId = I(2)

    sysHam = 0 * sysId
    envHam = 0 * envId
    if !isempty(locHam)
        sysHam = ManifestMatrix(locHam, sysOps)
        envHam = ManifestMatrix(locHam, envOps)
    end

    results = Dict("vals" => Float64[], "vecs" => Vector{Float64}[])

    @showprogress desc="bond dimension=$(bondDim)" for numSites in 1:maxSites
        sysOps = Dict(k => kron(v,  I(2)) for (k,v) in sysOps)
        merge!(sysOps, Dict((k, numSites+1) => kron(sysId, v) for (k,v) in InitOperators()))
        envOps = Dict(k => kron(I(2), v) for (k,v) in envOps)
        merge!(envOps, Dict((k, numSites+1) => kron(v, envId) for (k,v) in InitOperators()))

        sysHam = kron(sysHam, I(2)) + ManifestMatrix(incHam(numSites+1), sysOps)
        envHam = kron(I(2), envHam) + ManifestMatrix(incHam(numSites+1), envOps)

        sysId = kron(sysId, I(2))
        envId = kron(I(2), envId)

        superHam = kron(sysId, envHam) + kron(sysHam, envId)
        superHam += Glue(glueHam(numSites+1), sysOps, envOps)

        vals, vecs = eigen(Hermitian(superHam))
        push!(results["vals"], vals[1])
        push!(results["vecs"], vecs[:, 1])

        groundState = vecs[:, 1]
        groundStateTensor = reshape(groundState, (size(envHam)[1], size(sysHam)[1]))'
        F = svd(groundStateTensor)
        sysRotate = F.U[:, 1:minimum((bondDim, length(F.S)))]
        envRotate = F.V[:, 1:minimum((bondDim, length(F.S)))]

        sysId = sysRotate' * sysId * sysRotate
        envId = envRotate' * envId * envRotate
        sysHam = sysRotate' * sysHam * sysRotate
        envHam = envRotate' * envHam * envRotate
        sysOps = Dict(k => sysRotate' * op * sysRotate for (k, op) in sysOps)
        envOps = Dict(k => envRotate' * op * envRotate for (k, op) in envOps)
    end
    return results

end
