from labelpropagation.label_propagation import LabelPropagation

lp = LabelPropagation("input/weight", "W")
lp.run(label_resolution="retention", equilibrium="change", order="asynchronous", draw=True, include_weights=True)
lp.evaluate(label_resolution="retention", equilibrium="change", order="asynchronous", k=100, include_weights=True)
