using Test
using LinearAlgebra

include("dmrg.jl")  # adjust path if needed

global locHam = []
global incHam = i -> [("zz", [i-1, i], 0.25), ("+-", [i-1, i], 0.5), ("-+", [i-1, i], 0.5)]
global glueHam = i -> [("zz", [i, -i], 0.25), ("+-", [i, -i], 0.5), ("-+", [i, -i], 0.5)]
global bondDim = 23

# === helper: exact diagonalization for Heisenberg chain ===
function exact_heisenberg(N::Int)
    Sp = [0.0 1.0;
          0.0 0.0]
    Sm = [0.0 0.0;
          1.0 0.0]
    Sz = [0.5 0.0;
          0.0 -0.5]

    H = zeros(2^N, 2^N)

    for i in 1:(N-1)
        term = kron(Sz, Sz)
        term += 0.5 * kron(Sp, Sm)
        term += 0.5 * kron(Sm, Sp)
        H .+= kron(I(2^(i-1)), term, I(2^(N - i - 1)))
    end

    vals = eigen(Hermitian(H)).values
    return minimum(real(vals))
end

@testset "DMRG correctness tests" begin

    # --------------------------
    # 1. Small chains: compare ED
    # --------------------------
    #
    @testset "Exact diagonalization (small L)" begin
        for Nsteps in 1:4   # corresponds to 4,6,8 sites
            L = 2 + 2*Nsteps
            E_exact = exact_heisenberg(L)
            results = InfiniteDMRG(Nsteps, bondDim, locHam, incHam, glueHam)
            @test isapprox(results["vals"][end], E_exact; atol=1e-5)
        end
    end

    # --------------------------
    # 2. Bethe ansatz reference
    # --------------------------
    @testset "Bethe ansatz comparison" begin
        bethe_data = Dict(
            16 => -6.9117371455749,
            24 => -10.4537857604096,
            32 => -13.9973156182243,
            48 => -21.0859563143863,
            64 => -28.1754248597421,
        )

        Nsteps = (maximum(keys(bethe_data)) - 2) ÷ 2
        results = InfiniteDMRG(Nsteps, bondDim, locHam, incHam, glueHam)
        for (L, E_ref) in bethe_data
            Nsteps = (L - 2) ÷ 2
            @test isapprox(results["vals"][(L - 1) ÷ 2], E_ref; atol=1e-4)
        end
    end
end
