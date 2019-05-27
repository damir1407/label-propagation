from labelpropagation.label_propagation import LabelPropagation
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import time


def draw_graph(graph, label_map, title):
    colors = [label_map.get(node) for node in graph.nodes()]
    plt.subplot(111)
    nx.draw(graph, with_labels=graph.nodes, node_color=colors)
    plt.title(title)
    plt.show()


lp = LabelPropagation("data/test", "U")

start_time = time.time()
labels = lp.run(label_resolution="random", equilibrium="change", order="asynchronous", weighted=False)
print(time.time() - start_time)
print(len(Counter(labels.values())))
draw_graph(lp.graph, labels, "Label Propagation")

start_time = time.time()
labels = lp.consensus_clustering(label_resolution="retention", equilibrium="change", order="asynchronous",
                                 threshold=6, number_of_partitions=12, weighted=False)
print(time.time() - start_time)
print(len(Counter(labels.values())))
draw_graph(lp.graph, labels, "Consensus Clustering")
