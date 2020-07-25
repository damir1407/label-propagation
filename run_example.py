from labelpropagation.label_propagation import LabelPropagation
import networkx as nx
import time
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score


def community_modularity(coms, g):
    if type(g) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = g.size(weight='weight')
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in g:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.) + g.degree(node, weight='weight')
            for neighbor, dt in g[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.) + float(weight) / 2.
        except:
            pass

    res = 0.
    for com in set(coms.values()):
        res += (inc.get(com, 0.) / links) - (deg.get(com, 0.) / (2.*links))**2
    return res


def pquality_summary(g, coms):
    part = {}
    ids = 0
    largest_community_index = 0
    for i, c in enumerate(coms):
        if len(c) > len(coms[largest_community_index]):
            largest_community_index = i
        for n in c:
            part[n] = ids
        ids += 1

    mod = community_modularity(part, g)
    return mod, len(coms[largest_community_index]) / len(list(g))


def flake_score(network, node_labels):
    nodes = list(network)
    count_flake_nodes = 0
    for node in nodes:
        ki = len(network[node])
        same_label = 0
        for v in network[node]:
            if node_labels[node] == node_labels[v]:
                same_label += 1

        if same_label < ki / 2:
            count_flake_nodes += 1

    return count_flake_nodes / len(nodes)


def largest_community_share(network, found_communities):
    largest_community_index = 0
    for i, c in enumerate(found_communities):
        if len(c) > len(found_communities[largest_community_index]):
            largest_community_index = i

    return len(found_communities[largest_community_index]) / len(list(network))


for file in ["data/arenas-pgp/out.arenas-pgp", "data/douban/out.douban", "data/com-youtube/out.com-youtube"]:
    break

d = 10
no_of_nodes = [100, 1000, 10000, 100000, 1000000]
# no_of_nodes = [100]
for n in no_of_nodes:

    lpa_time = []
    lpa_iterations = []
    lpa_communities = []
    lpa_modularity = []
    lpa_flake = []
    lpa_share = []
    cc_time = []
    cc_iterations = []
    cc_communities = []
    cc_modularity = []
    fcc_time = []
    fcc_iterations = []
    fcc_communities = []
    fcc_modularity = []
    print("NUMBER OF NODES:", n)
    m = int((n*d)/2)
    print("NUMBER OF EDGES:", m)

    for i in range(0, 10):
        G = nx.gnm_random_graph(n, m)
        print("Graph ready", i)
        lp = LabelPropagation(network=G)
        _, _ = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous",
                        weighted=False)
        print("Algorithm ready", i)
        start_time = time.time()
        lpa_time.append(lp.method_time)
        print("LPA TIME:", time.time() - start_time)
        start_time = time.time()
        lpa_iterations.append(lp.iterations)
        print("LPA ITERATIONS:", time.time() - start_time)
        start_time = time.time()
        lpa_communities.append(lp.number_of_communities)
        print("LPA COMMUNITIES:", time.time() - start_time)
        start_time = time.time()
        lpa_flake.append(flake_score(G, lp.node_labels))
        print("LPA FLAKE:", time.time() - start_time)
        # start_time = time.time()
        # mod, share = pquality_summary(G, list(lp.final_communities))
        # lpa_modularity.append(mod)
        # lpa_share.append(share)
        # print("LPA MOD SHARE:", time.time() - start_time)
        start_time = time.time()
        lpa_share.append(largest_community_share(G, list(lp.final_communities)))
        print("LPA SHARE:", time.time() - start_time)
        start_time = time.time()
        lpa_modularity.append(nx.algorithms.community.modularity(G, lp.final_communities))
        print("LPA BUILT IN MOD:", time.time() - start_time)
        print("Metrics ready", i)

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
    print("LPA MODULARITY", lpa_modularity)
    print("LPA AVG MODULARITY", np.mean(lpa_modularity))
    print("LPA STD MODULARITY", np.std(lpa_modularity))
    print("LPA FLAKE", lpa_flake)
    print("LPA AVG FLAKE", np.mean(lpa_flake))
    print("LPA STD FLAKE", np.std(lpa_flake))
    print("LPA LARGEST COMMUNITY SHARE", lpa_share)
    print("LPA AVG LARGEST COMMUNITY SHARE", np.mean(lpa_share))
    print("LPA STD LARGEST COMMUNITY SHARE", np.std(lpa_share))
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

for n in no_of_nodes:

    lpa_time = []
    lpa_iterations = []
    lpa_communities = []
    lpa_modularity = []
    lpa_flake = []
    lpa_share = []
    cc_time = []
    cc_iterations = []
    cc_communities = []
    cc_modularity = []
    fcc_time = []
    fcc_iterations = []
    fcc_communities = []
    fcc_modularity = []
    print("NUMBER OF NODES:", n)
    m = int(d / 2)
    print("NUMBER OF EDGES FOR EACH NODE:", m)

    for i in range(0, 10):
        G = nx.barabasi_albert_graph(n, m)
        print("Graph ready", i)
        lp = LabelPropagation(network=G)
        _, _ = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous",
                        weighted=False)
        print("Algorithm ready", i)
        start_time = time.time()
        lpa_time.append(lp.method_time)
        print("LPA TIME:", time.time() - start_time)
        start_time = time.time()
        lpa_iterations.append(lp.iterations)
        print("LPA ITERATIONS:", time.time() - start_time)
        start_time = time.time()
        lpa_communities.append(lp.number_of_communities)
        print("LPA COMMUNITIES:", time.time() - start_time)
        start_time = time.time()
        lpa_flake.append(flake_score(G, lp.node_labels))
        print("LPA FLAKE:", time.time() - start_time)
        start_time = time.time()
        mod, share = pquality_summary(G, list(lp.final_communities))
        lpa_modularity.append(mod)
        lpa_share.append(share)
        print("LPA MOD SHARE:", time.time() - start_time)
        print("Metrics ready", i)

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
    print("LPA MODULARITY", lpa_modularity)
    print("LPA AVG MODULARITY", np.mean(lpa_modularity))
    print("LPA STD MODULARITY", np.std(lpa_modularity))
    print("LPA FLAKE", lpa_flake)
    print("LPA AVG FLAKE", np.mean(lpa_flake))
    print("LPA STD FLAKE", np.std(lpa_flake))
    print("LPA LARGEST COMMUNITY SHARE", lpa_share)
    print("LPA AVG LARGEST COMMUNITY SHARE", np.mean(lpa_share))
    print("LPA STD LARGEST COMMUNITY SHARE", np.std(lpa_share))
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

for n in no_of_nodes:

    lpa_time = []
    lpa_iterations = []
    lpa_communities = []
    lpa_modularity = []
    lpa_flake = []
    lpa_share = []
    cc_time = []
    cc_iterations = []
    cc_communities = []
    cc_modularity = []
    fcc_time = []
    fcc_iterations = []
    fcc_communities = []
    fcc_modularity = []
    print("NUMBER OF NODES:", n)
    m = int(d / 2)
    print("NUMBER OF EDGES FOR EACH NODE:", m)

    for i in range(0, 10):
        params = {"n": n, "tau1": 3, "tau2": 2, "mu": 0.5, "average_degree": d, "max_degree": d}
        start_time = time.time()
        G = nx.LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], params["mu"],
                                min_degree=params["average_degree"],
                                max_degree=params["max_degree"],
                                max_iters=5000,
                                )
        communities = {frozenset(G.nodes[v]['community']) for v in G}
        # print(len(communities))
        communities = [list(c) for c in communities]
        # print(len(communities))
        # print(communities)
        print("Graph ready", i, time.time() - start_time)
        lp = LabelPropagation(network=G)
        _, _ = lp.start(label_ties_resolution="retention", convergence_criterium="change", order="asynchronous",
                        weighted=False)
        # print("Algorithm ready", i)
        # print(lp.final_communities)
        # print(normalized_mutual_info_score(communities, list(lp.final_communities)))
        start_time = time.time()
        lpa_time.append(lp.method_time)
        print("LPA TIME:", time.time() - start_time)
        start_time = time.time()
        lpa_iterations.append(lp.iterations)
        print("LPA ITERATIONS:", time.time() - start_time)
        start_time = time.time()
        lpa_communities.append(lp.number_of_communities)
        print("LPA COMMUNITIES:", time.time() - start_time)
        start_time = time.time()
        lpa_flake.append(flake_score(G, lp.node_labels))
        print("LPA FLAKE:", time.time() - start_time)
        # start_time = time.time()
        # mod, share = pquality_summary(G, list(lp.final_communities))
        # lpa_modularity.append(mod)
        # lpa_share.append(share)
        # print("LPA MOD SHARE:", time.time() - start_time)
        start_time = time.time()
        lpa_share.append(largest_community_share(G, list(lp.final_communities)))
        print("LPA SHARE:", time.time() - start_time)
        start_time = time.time()
        lpa_modularity.append(nx.algorithms.community.modularity(G, lp.final_communities))
        print("LPA BUILT IN MOD:", time.time() - start_time)
        print("Metrics ready", i)
    print("LPA TIME", lpa_time)
    print("LPA MEAN TIME", np.mean(lpa_time))
    print("LPA STD TIME", np.std(lpa_time))
    print("LPA ITERATIONS", lpa_iterations)
    print("LPA AVG ITERATIONS", np.mean(lpa_iterations))
    print("LPA STD ITERATIONS", np.std(lpa_iterations))
    print("LPA COMMUNITIES", lpa_communities)
    print("LPA AVG COMMUNITIES", np.mean(lpa_communities))
    print("LPA STD COMMUNITIES", np.std(lpa_communities))
    print("LPA MODULARITY", lpa_modularity)
    print("LPA AVG MODULARITY", np.mean(lpa_modularity))
    print("LPA STD MODULARITY", np.std(lpa_modularity))
    print("LPA FLAKE", lpa_flake)
    print("LPA AVG FLAKE", np.mean(lpa_flake))
    print("LPA STD FLAKE", np.std(lpa_flake))
    print("LPA LARGEST COMMUNITY SHARE", lpa_share)
    print("LPA AVG LARGEST COMMUNITY SHARE", np.mean(lpa_share))
    print("LPA STD LARGEST COMMUNITY SHARE", np.std(lpa_share))
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