from labelpropagation.label_propagation import LabelPropagation
import networkx as nx
from networkx.algorithms.community import label_propagation_communities, asyn_lpa_communities
import time

no_of_nodes = [(100, 0.1), (1000, 0.1), (10000, 0.05), (100000, 0.01), (1000000, 0.0001)]
for n in no_of_nodes:
    G = nx.erdos_renyi_graph(n[0], n[1])
    print("ER graph ready")
    lp = LabelPropagation(network=G)
    _, _ = lp.start(label_ties_resolution="random", convergence_criterium="change", order="asynchronous",
                    weighted=False)
    start_time = time.time()
    _ = asyn_lpa_communities(G)
    print("NX TIME:", time.time() - start_time)
    print("TIME:", lp.method_time)
    print("NUMBER OF COMMUNITIES:", lp.number_of_communities)
    print("NUMBER OF ITERATIONS:", lp.iterations)
    print()

# network, labels = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="strong-community", order="asynchronous", threshold=0.5, number_of_partitions=12, weighted=False, fcc=False)
#
# network, labels = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="strong-community", order="asynchronous", threshold=0.5, number_of_partitions=12, weighted=False, fcc=True, convergence_factor=0.02)
#
