from labelpropagation import LabelPropagation

lp = LabelPropagation("input/inputFile")
lp.run("random")
lp.evaluate("random", 100)
