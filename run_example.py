from labelpropagation.label_propagation import LabelPropagation
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(graph, label_map, title):
    colors = [label_map.get(node) for node in graph.nodes()]
    plt.subplot(111)
    nx.draw(graph, with_labels=graph.nodes, node_color=colors)
    plt.title(title)
    plt.show()


lp = LabelPropagation("data/moreno_blogs/out.moreno_blogs_blogs", "U")

labels = lp.run(label_resolution="retention", equilibrium="change", order="asynchronous", weighted=False)
print(labels)
draw_graph(lp.graph, labels, "Label Propagation")

labels = lp.consensus_clustering(label_resolution="retention", equilibrium="change", order="asynchronous",
                                 threshold=6, number_of_partitions=12, weighted=False)
print(labels)
draw_graph(lp.graph, labels, "Consensus Clustering")
