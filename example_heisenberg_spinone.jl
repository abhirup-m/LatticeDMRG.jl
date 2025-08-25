using Plots

include("dmrg.jl")

locHam = [("z", [1], 0.)]
incHam = i -> [("zz", [i-1, i], 1.0), ("+-", [i-1, i], 0.5), ("-+", [i-1, i], 0.5)]
glueHam = i -> [("zz", [i, -i], 1.0), ("+-", [i, -i], 0.5), ("-+", [i, -i], 0.5)]
results = InfiniteDMRG(21, 20, locHam, incHam, glueHam; correlation=Dict("Sz-$(i)" => [("z", [i], 1.0)] for i in 1:20), spin=1.)
display(results["vals"])
display(results["vecs"][end])
display([results["Sz-$(i)"] for i in 1:20])
p = Plots.plot([abs(results["Sz-$(i)"]) for i in 1:20])
