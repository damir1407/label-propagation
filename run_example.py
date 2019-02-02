from labelpropagation.label_propagation import LabelPropagation
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(graph, label_map):
    colors = [label_map.get(node) for node in graph.nodes()]
    plt.subplot(111)
    nx.draw(graph, with_labels=graph.nodes, node_color=colors)
    plt.show()


lp = LabelPropagation("data/dolphins/out.dolphins", "U")

labels = lp.run(label_resolution="retention", equilibrium="change", order="asynchronous", weighted=False)
draw_graph(lp.graph, labels)

# TODO: Check if subnetworks of same label appear
print(lp.graph.__dict__["_adj"])
community_set = set(labels.values())
community_dict = {value: [] for value in community_set}
for key, value in labels.items():
    community_dict[value].append(key)
print(community_dict)
print(Counter(labels.values()))
print(len(Counter(labels.values())))

labels = lp.consensus_clustering(label_resolution="retention", equilibrium="change", order="asynchronous",
                                 threshold=6, number_of_partitions=12, weighted=False)
draw_graph(lp.graph, labels)
