include("dmrg.jl")

results = InfiniteDMRG(10, 20)
display(results["vals"])
display(results["vecs"][end])
