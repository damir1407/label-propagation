from labelpropagation import LabelPropagation

lp = LabelPropagation("input/inputFile")
lp.run("random", "raghavan", True)
lp.evaluate("random", "raghavan", 100)
