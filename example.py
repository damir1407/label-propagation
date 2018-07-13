from labelpropagation.label_propagation import LabelPropagation

lp = LabelPropagation("input/weight", "W")
lp.run("weight", "retention", "change", "asynchronous", True)
lp.evaluate("weight", "retention", "change", "asynchronous", 100)
