from labelpropagation import LabelPropagation

lp = LabelPropagation("input/weight", "W")
lp.run("retention", "strong-community", "asynchronous", True)
lp.evaluate("retention", "strong-community", "asynchronous", 100)
