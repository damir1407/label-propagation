from labelpropagation import LabelPropagation

lp = LabelPropagation("input/inputFile")
lp.run("random", "strong-community", "asynchronous", True)
lp.evaluate("random", "strong-community", "asynchronous", 100)
