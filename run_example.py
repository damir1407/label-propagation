from labelpropagation.label_propagation import LabelPropagation

lp = LabelPropagation("path/to/data.txt", "U")

network, labels = lp.start(label_ties_resolution="retention", convergence_criterium="strong-community", order="asynchronous", weighted=False)

network, labels = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="strong-community", order="asynchronous", threshold=6, number_of_partitions=12, weighted=False, fcc=False)

network, labels = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="strong-community", order="asynchronous", threshold=6, number_of_partitions=12, weighted=False, fcc=True)

