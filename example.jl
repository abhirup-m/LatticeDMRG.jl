include("dmrg.jl")

locHam = [("z", [1], 0.)]
incHam = i -> [("zz", [i-1, i], 0.25), ("+-", [i-1, i], 0.5), ("-+", [i-1, i], 0.5)]
glueHam = i -> [("zz", [i, -i], 0.25), ("+-", [i, -i], 0.5), ("-+", [i, -i], 0.5)]
results = InfiniteDMRG(10, 20, locHam, incHam, glueHam)
display(results["vals"])
display(results["vecs"][end])
