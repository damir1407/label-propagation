from labelpropagation import LabelPropagation

lp = LabelPropagation("input/weight", "W")
lp.run("label", "retention", "strong-community", "asynchronous", True)
lp.evaluate("label", "retention", "strong-community", "asynchronous", 100)
