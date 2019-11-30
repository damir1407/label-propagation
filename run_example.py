from labelpropagation.label_propagation import LabelPropagation
from collections import Counter

lp = LabelPropagation("data/dolphins/out.dolphins", "U")

_, labels = lp.run(label_resolution="retention", equilibrium="strong-community", order="asynchronous", weighted=False)
print(len(Counter(labels.values())))

_, labels = lp.consensus_clustering(label_resolution="retention", equilibrium="strong-community", order="asynchronous",
                                 threshold=6, number_of_partitions=12, weighted=False)
print(len(Counter(labels.values())))

