from itertools import combinations
from collections import Counter
import networkx as nx
import random
import copy
import numpy as np
import time


class LabelPropagation:
    def __init__(self, file_path=None, network=None):
        self._weight = None
        if file_path:
            self.network = self._initialize_graph_from_file(file_path)
        else:
            self.network = network
        self.node_labels = {}
        self._settings = {}
        self._seed = 2
        self.recursive_steps = None
        self.iterations = None
        self.cc_iterations = None
        self.method_time = None
        self.number_of_communities = None
        self.final_communities = None

    def _initialize_graph_from_file(self, file_path):
        new_graph = nx.Graph()
        with open(file_path, "rt") as f:
            edges = []
            for line in f:
                line = line.split()
                if len(line) < 3:
                    edges.append((line[0], line[1]))
                else:
                    edges.append((line[0], line[1], int(line[2])))
            if len(line) < 3:
                new_graph.add_edges_from(edges)
            else:
                new_graph.add_weighted_edges_from(edges)
                self._weight = "weight"
        return new_graph

    def start(self, label_ties_resolution, convergence_criterium, order, weighted=False, maximum_iterations=100):
        self._settings = {
            "label_ties_resolution": label_ties_resolution,
            "convergence_criterium": convergence_criterium,
            "order_of_label_propagation": order,
            "weighted": weighted,
            "maximum_iterations": maximum_iterations,
        }

        start_time = time.time()
        self._main()
        self.method_time = time.time() - start_time

        communities = self._get_communities()
        self.number_of_communities = len(communities)
        if len(communities) > 1:
            self.final_communities = self._dfs_connectivity(communities)
        else:
            self.final_communities = communities
        if len(self.final_communities) > len(communities):
            print("Disconnected communities found!")
            for index, community in enumerate(self.final_communities):
                for member in community:
                    self.node_labels[member] = index

        return self.network, self.node_labels

    def _get_communities(self):
        unique_community_labels = set(self.node_labels.values())
        unique_communities_by_label = {value: [] for value in unique_community_labels}
        for node, label in self.node_labels.items():
            unique_communities_by_label[label].append(node)
        return unique_communities_by_label.values()

    @staticmethod
    def _random(node, max_labels):
        return random.choice(max_labels)

    @staticmethod
    def _inclusion(node, max_labels):
        return random.choice(max_labels)

    def _retention(self, node, max_labels):
        if self.node_labels[node] in max_labels:
            return self.node_labels[node]
        return random.choice(max_labels)

    def _find_max_labels_in_neighborhood(self, node):
        label_freq = Counter()
        for v in self.network[node]:
            label_freq.update({self.node_labels[v]: self.network.edges[node, v][self._weight] if self._weight else 1})
        if self._settings["label_ties_resolution"] == "inclusion":
            label_freq.update({self.node_labels[node]: 1})

        max_freq = max(label_freq.values())
        return [label for label, freq in label_freq.items() if freq == max_freq]

    def _maximal_neighbouring_label(self, node):
        best_labels = self._find_max_labels_in_neighborhood(node)

        if len(best_labels) == 1:
            return best_labels[0]
        else:
            try:
                return eval("self._" + self._settings["label_ties_resolution"] + "(node, best_labels)")
            except Exception:
                raise ValueError("Invalid label ties resolution parameter. Choose between \"random\", \"inclusion\" and \"retention\".")

    def _dfs_recursive(self, v, visited, community, connected_group):
        visited[v] = True

        for i in self.network[v].keys():
            if i in community and not visited[i]:
                community.remove(i)
                connected_group.append(i)
                connected_group = self._dfs_recursive(i, visited, community, connected_group)
        return connected_group

    def _dfs_connectivity(self, communities):
        result = []

        for community in communities:
            community_copy = copy.deepcopy(community)
            visited = {member: False for member in community}
            while len(community_copy) != 0:
                v = random.choice(community_copy)
                community_copy.remove(v)
                result.append(self._dfs_recursive(v, visited, community_copy, [v]))
        return result

    def _convergence(self):
        nodes = list(self.network)
        for node in nodes:
            max_labels = self._find_max_labels_in_neighborhood(node)

            if len(max_labels) > 1:
                if self._settings["convergence_criterium"] == "label-equilibrium":
                    if self.node_labels[node] in max_labels:
                        continue
                    else:
                        return True
                elif self._settings["convergence_criterium"] == "strong-community":
                    return True
                else:
                    raise ValueError("Invalid label equilibrium criteria parameter. Choose between \"label-equilibrium\", \"strong-community\" and \"change\".")
            elif self.node_labels[node] != max_labels[0]:
                return True
        return False

    def _asynchronous_propagation(self):
        change = False
        nodes = list(self.network)
        # random.seed(self._seed)
        random.shuffle(nodes)

        for node in nodes:
            if len(self.network[node]) < 1:
                self.network.add_edge(node, random.choice(nodes))
                # continue

            new_label = self._maximal_neighbouring_label(node)

            if self.node_labels[node] != new_label:
                self.node_labels[node] = new_label
                change = True

        return change

    def _synchronous_propagation(self):
        change = False
        nodes = list(self.network)
        # random.seed(self._seed)
        random.shuffle(nodes)
        sync_label_map = copy.deepcopy(self.node_labels)

        for node in nodes:
            if len(self.network[node]) < 1:
                continue

            new_label = self._maximal_neighbouring_label(node)

            if sync_label_map[node] != new_label:
                sync_label_map[node] = new_label
                change = True

        self.node_labels = sync_label_map
        return change

    def _main(self):
        self.node_labels = {node: label for label, node in enumerate(self.network)}
        change = True
        self.iterations = 0
        while change and self.iterations < self._settings["maximum_iterations"]:
            self.iterations += 1
            try:
                change = eval("self._" + self._settings["order_of_label_propagation"] + "_propagation()")
            except Exception:
                raise ValueError("Invalid iteration order parameter")

            if self._settings["convergence_criterium"] != "change":
                change = self._convergence()

    def consensus_clustering(self, label_ties_resolution, convergence_criterium, order, threshold, number_of_partitions, max_recursive_steps=10, weighted=False, maximum_iterations=100, fcc=False, convergence_factor=0.02):
        if threshold > number_of_partitions or threshold < 0:
            raise ValueError("Threshold must be between 0 and number of partitions")

        self._settings = {
            "label_ties_resolution": label_ties_resolution,
            "convergence_criterium": convergence_criterium,
            "order_of_label_propagation": order,
            "threshold": threshold,
            "maximum_iterations": maximum_iterations,
            "weighted": weighted,
            "number_of_partitions": number_of_partitions,
            "max_recursive_steps": max_recursive_steps,
            "fcc": fcc,
            "convergence_factor": convergence_factor,
        }

        start_time = time.time()
        found_communities = {}
        self.cc_iterations = 0
        for i in range(self._settings["number_of_partitions"]):
            self._main()
            self.cc_iterations += self.iterations
            found_communities[i] = self.node_labels

        self._settings["weighted"] = True
        self.recursive_steps = 0
        self._recursive(found_communities)
        self.number_of_communities = nx.number_connected_components(self.network)
        self.final_communities = []
        for i, c in enumerate(nx.connected_components(self.network)):
            comm = list(c)
            self.final_communities.append(comm)
            for v in comm:
                self.node_labels[v] = i
        self.method_time = time.time() - start_time
        return self.network, self.node_labels

    def _recursive(self, found_communities):
        if self.recursive_steps >= self._settings["max_recursive_steps"]:
            print("Reached maximum recursive steps")
            return

        for u, v in self.network.edges():
            self.network[u][v]['weight'] = 0.0

        for node, nbr in self.network.edges():
            for i in range(self._settings["number_of_partitions"]):
                communities = found_communities[i]
                if communities[node] == communities[nbr]:
                    if not self.network.has_edge(node, nbr):
                        self.network.add_edge(node, nbr, weight=0)
                    self.network[node][nbr]['weight'] += 1

        edges_to_be_removed = []
        for u, v in self.network.edges():
            if self.network[u][v]['weight'] < self._settings["threshold"] * self._settings["number_of_partitions"]:
                edges_to_be_removed.append((u, v))

        self.network.remove_edges_from(edges_to_be_removed)

        if self._settings["fcc"]:
            for _ in range(self.network.number_of_edges()):
                node = np.random.choice(self.network.nodes())
                neighbors = [n[1] for n in self.network.edges(node)]

                if len(neighbors) >= 2:
                    j, k = random.sample(set(neighbors), 2)

                    if not self.network.has_edge(j, k):
                        self.network.add_edge(j, k, weight=0)

                        for i in range(self._settings["number_of_partitions"]):
                            communities = found_communities[i]
                            if communities[j] == communities[k]:
                                self.network[j][k]['weight'] += 1

        if self._is_block_diagonal():
            return

        found_communities = {}
        for i in range(self._settings["number_of_partitions"]):
            self._main()
            self.cc_iterations += self.iterations
            found_communities[i] = self.node_labels

        self.recursive_steps += 1
        self._recursive(found_communities)
        return

    def _is_block_diagonal(self):
        count_faulty_edges = 0
        for weight in nx.get_edge_attributes(self.network, 'weight').values():
            if self._settings["threshold"] < weight < self._settings["number_of_partitions"]:
                if self._settings["fcc"]:
                    count_faulty_edges += 1
                else:
                    return False

        if count_faulty_edges > self._settings["convergence_factor"] * self.network.number_of_edges():
            return False

        return True

    @staticmethod
    def _get_clean_d_matrix(input_matrix):
        d = {}
        for key, val in input_matrix.items():
            d[key] = {}
            for key2, val2 in val.items():
                d[key][key2] = {}
                d[key][key2]["weight"] = 0
        return d

    @staticmethod
    def _compute_d_matrix(d_matrix, found_communities):
        d_copy = copy.deepcopy(d_matrix)
        for community in found_communities:
            for members in community:
                for node1, node2 in combinations(members, r=2):
                    if node2 not in d_copy[node1]:
                        d_copy[node1][node2] = {"weight": 1}
                        d_copy[node2][node1] = {"weight": 1}
                    else:
                        d_copy[node1][node2]["weight"] += 1
                        d_copy[node2][node1]["weight"] += 1
        return d_copy
