from labelpropagation.label_propagation import LabelPropagation

lp = LabelPropagation("input-examples/weight", "W")
lp.run(label_resolution="retention", equilibrium="change", order="asynchronous", draw=False, include_weights=True)
lp.consensus_clustering(label_resolution="retention", equilibrium="change", order="asynchronous")
