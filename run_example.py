from labelpropagation.label_propagation import LabelPropagation

lp = LabelPropagation("path/to/data.txt", "U")

network, labels = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous", weighted=False)

network, labels = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous", threshold=0.5, number_of_partitions=10, weighted=False, fcc=False)

network, labels = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous", threshold=0.5, number_of_partitions=10, weighted=False, fcc=True, convergence_factor=0.02)
