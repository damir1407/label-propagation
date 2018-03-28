from labelpropagation import LabelPropagation

lp = LabelPropagation("input/inputFile")
lp.run("random", True)
lp.evaluate("random", 100)
