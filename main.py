from labelpropagation import LabelPropagation

lp = LabelPropagation("input/inputFile")
lp.run("random", "label-equilibrium", True)
lp.evaluate("random", "label-equilibrium", 100)
