from labelpropagation.label_propagation import LabelPropagation
import networkx as nx
from networkx.algorithms.community import label_propagation_communities, asyn_lpa_communities
import time
import numpy as np

for file in ["data/arenas-pgp/out.arenas-pgp", "data/douban/out.douban", "data/com-youtube/out.com-youtube"]:
    # start_time = time.time()

    # print("NUMBER OF NODES:", len(G.nodes))
    # print("NUMBER OF EDGES:", len(G.edges))
    # print("FAST ER TIME:", time.time() - start_time)

    lpa_time = []
    lpa_iterations = []
    lpa_communities = []
    cc_time = []
    cc_iterations = []
    cc_communities = []
    fcc_time = []
    fcc_iterations = []
    fcc_communities = []

    for i in range(0, 10):
        lp = LabelPropagation(file_path=file)
        _, _ = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous",
                        weighted=False)
        lpa_time.append(lp.method_time)
        lpa_iterations.append(lp.iterations)
        lpa_communities.append(lp.number_of_communities)

        # lp = LabelPropagation(network=G)
        # _, _ = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="change",
        #                                order="asynchronous", threshold=0.5, number_of_partitions=10, weighted=False,
        #                                fcc=False)
        # cc_time.append(lp.method_time)
        # cc_iterations.append(lp.recursive_steps)
        # cc_communities.append(lp.number_of_communities)
        #
        # lp = LabelPropagation(network=G)
        # _, _ = lp.consensus_clustering(label_ties_resolution="retention",
        #                                convergence_criterium="change", order="asynchronous",
        #                                threshold=0.5, number_of_partitions=10, weighted=False, fcc=True,
        #                                convergence_factor=0.02)
        # fcc_time.append(lp.method_time)
        # fcc_iterations.append(lp.recursive_steps)
        # fcc_communities.append(lp.number_of_communities)
    print("LPA TIME", lpa_time)
    print("LPA MEAN TIME", np.mean(lpa_time))
    print("LPA STD TIME", np.std(lpa_time))
    print("LPA ITERATIONS", lpa_iterations)
    print("LPA AVG ITERATIONS", np.mean(lpa_iterations))
    print("LPA STD ITERATIONS", np.std(lpa_iterations))
    print("LPA COMMUNITIES", lpa_communities)
    print("LPA AVG COMMUNITIES", np.mean(lpa_communities))
    print("LPA STD COMMUNITIES", np.std(lpa_communities))
    # print("CC TIME", cc_time)
    # print("CC MEAN TIME", np.mean(cc_time))
    # print("CC STD TIME", np.std(cc_time))
    # print("CC RECURSIVE STEPS", cc_iterations)
    # print("CC AVG RECURSIVE STEPS", np.mean(cc_iterations))
    # print("CC STD RECURSIVE STEPS", np.std(cc_iterations))
    # print("CC COMMUNITIES", cc_communities)
    # print("CC AVG COMMUNITIES", np.mean(cc_communities))
    # print("CC STD COMMUNITIES", np.std(cc_communities))
    # print("FCC TIME", fcc_time)
    # print("FCC MEAN TIME", np.mean(fcc_time))
    # print("FCC STD TIME", np.std(fcc_time))
    # print("FCC RECURSIVE STEPS", fcc_iterations)
    # print("FCC AVG RECURSIVE STEPS", np.mean(fcc_iterations))
    # print("FCC STD RECURSIVE STEPS", np.std(fcc_iterations))
    # print("FCC COMMUNITIES", fcc_communities)
    # print("FCC AVG COMMUNITIES", np.mean(fcc_communities))
    # print("FCC STD COMMUNITIES", np.std(fcc_communities))
    print("==============================================")
    print()

no_of_nodes = [(100, 0.05), (1000, 0.005), (10000, 0.0005), (100000, 0.00005), (1000000, 0.000005)]
for n in no_of_nodes:
    # start_time = time.time()

    # print("NUMBER OF NODES:", len(G.nodes))
    # print("NUMBER OF EDGES:", len(G.edges))
    # print("FAST ER TIME:", time.time() - start_time)

    lpa_time = []
    lpa_iterations = []
    lpa_communities = []
    cc_time = []
    cc_iterations = []
    cc_communities = []
    fcc_time = []
    fcc_iterations = []
    fcc_communities = []
    print("NUMBER OF NODES:", n[0])

    for i in range(0, 10):
        G = nx.fast_gnp_random_graph(n[0], n[1])
        # G = nx.LFR_benchmark_graph()
        lp = LabelPropagation(network=G)
        _, _ = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous",
                        weighted=False)
        lpa_time.append(lp.method_time)
        lpa_iterations.append(lp.iterations)
        lpa_communities.append(lp.number_of_communities)

        # lp = LabelPropagation(network=G)
        # _, _ = lp.consensus_clustering(label_ties_resolution="retention", convergence_criterium="change",
        #                                order="asynchronous", threshold=0.5, number_of_partitions=10, weighted=False,
        #                                fcc=False)
        # cc_time.append(lp.method_time)
        # cc_iterations.append(lp.recursive_steps)
        # cc_communities.append(lp.number_of_communities)
        #
        # lp = LabelPropagation(network=G)
        # _, _ = lp.consensus_clustering(label_ties_resolution="retention",
        #                                convergence_criterium="change", order="asynchronous",
        #                                threshold=0.5, number_of_partitions=10, weighted=False, fcc=True,
        #                                convergence_factor=0.02)
        # fcc_time.append(lp.method_time)
        # fcc_iterations.append(lp.recursive_steps)
        # fcc_communities.append(lp.number_of_communities)
    print("LPA TIME", lpa_time)
    print("LPA MEAN TIME", np.mean(lpa_time))
    print("LPA STD TIME", np.std(lpa_time))
    print("LPA ITERATIONS", lpa_iterations)
    print("LPA AVG ITERATIONS", np.mean(lpa_iterations))
    print("LPA STD ITERATIONS", np.std(lpa_iterations))
    print("LPA COMMUNITIES", lpa_communities)
    print("LPA AVG COMMUNITIES", np.mean(lpa_communities))
    print("LPA STD COMMUNITIES", np.std(lpa_communities))
    # print("CC TIME", cc_time)
    # print("CC MEAN TIME", np.mean(cc_time))
    # print("CC STD TIME", np.std(cc_time))
    # print("CC RECURSIVE STEPS", cc_iterations)
    # print("CC AVG RECURSIVE STEPS", np.mean(cc_iterations))
    # print("CC STD RECURSIVE STEPS", np.std(cc_iterations))
    # print("CC COMMUNITIES", cc_communities)
    # print("CC AVG COMMUNITIES", np.mean(cc_communities))
    # print("CC STD COMMUNITIES", np.std(cc_communities))
    # print("FCC TIME", fcc_time)
    # print("FCC MEAN TIME", np.mean(fcc_time))
    # print("FCC STD TIME", np.std(fcc_time))
    # print("FCC RECURSIVE STEPS", fcc_iterations)
    # print("FCC AVG RECURSIVE STEPS", np.mean(fcc_iterations))
    # print("FCC STD RECURSIVE STEPS", np.std(fcc_iterations))
    # print("FCC COMMUNITIES", fcc_communities)
    # print("FCC AVG COMMUNITIES", np.mean(fcc_communities))
    # print("FCC STD COMMUNITIES", np.std(fcc_communities))
    print("==============================================")
    print()


    # start_time = time.time()
    # G = nx.barabasi_albert_graph(n[0], 2)
    # print("NUMBER OF EDGES:", len(G.edges))
    # print("FAST ER TIME:", time.time() - start_time)
    # print("ER graph ready")
    # lp = LabelPropagation(network=G)
    # _, _ = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous",
    #                 weighted=False)
    # start_time = time.time()
    # _ = asyn_lpa_communities(G)
    # print("NX TIME:", time.time() - start_time)
    # print("TIME:", lp.method_time)
    # print("NUMBER OF COMMUNITIES:", lp.number_of_communities)
    # print("NUMBER OF ITERATIONS:", lp.iterations)
    # print("===================================================")
