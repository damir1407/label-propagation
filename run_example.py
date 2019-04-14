from labelpropagation.label_propagation import LabelPropagation
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(graph, label_map):
    colors = [label_map.get(node) for node in graph.nodes()]
    plt.subplot(111)
    nx.draw(graph, with_labels=graph.nodes, node_color=colors)
    plt.show()


def get_communities(labels):
    community_set = set(labels.values())
    community_dict = {value: [] for value in community_set}
    for key, value in labels.items():
        community_dict[value].append(key)
    return community_dict.values()


lp = LabelPropagation("data/test", "U")

# labels = lp.run(label_resolution="inclusion", equilibrium="label-equilibrium", order="asynchronous", weighted=False)
# print(labels)
labels = {'1': 0, '2': 0, '3': 0, '4': 1, '5': 0, '6': 0, '7': 0, '8': 1, '9': 1, '10': 1}
draw_graph(lp.graph, labels)

communities = get_communities(labels)
result = lp.dfs(communities)
if len(result) > len(communities):
    for index, community in enumerate(result):
        for member in community:
            labels[member] = index
print(labels)
draw_graph(lp.graph, labels)

"""
labels = lp.consensus_clustering(label_resolution="retention", equilibrium="change", order="asynchronous",
                                 threshold=6, number_of_partitions=12, weighted=False)
draw_graph(lp.graph, labels)
"""