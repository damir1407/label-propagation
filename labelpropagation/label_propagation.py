from itertools import combinations
import networkx as nx
import random
import copy
import warnings
import numpy as np


class LabelPropagation:
    def __init__(self, file_path=None, network_type="U", network=None):
        self._network_type = network_type
        if file_path:
            self._network = self._initialize_graph_from_file(file_path)
        else:
            self._network = network
        self._node_labels = {}
        self._settings = {}
        self.recursive_steps = None
        self.iterations = None

    def _initialize_graph_from_file(self, file_path):
        new_graph = nx.Graph()
        with open(file_path, "rt") as f:
            edges = []
            for line in f:
                line = line.split()
                if self._network_type == "W" and len(line) < 3:
                    raise ValueError("Input file does not contain weights.")
                if self._network_type == "U":
                    edges.append((line[0], line[1]))
                elif self._network_type == "W":
                    edges.append((line[0], line[1], int(line[2])))
            if self._network_type == "U":
                new_graph.add_edges_from(edges)
            elif self._network_type == "W":
                new_graph.add_weighted_edges_from(edges)
        return new_graph

    def start(self, label_ties_resolution, convergence_criterium, order, weighted=False, maximum_iterations=100):
        if weighted and not self._network_type == "W":
            raise ValueError("Cannot perform label propagation that includes weighted edges, because graph type is not \"W\" (Weighted)")
        if weighted and label_ties_resolution == "inclusion":
            warnings.warn("Inclusion cannot be used on weighted graphs, random resolution will be performed instead")

        self._settings = {
            "label_ties_resolution": label_ties_resolution,
            "convergence_criterium": convergence_criterium,
            "order_of_label_propagation": order,
            "weighted": weighted,
            "maximum_iterations": maximum_iterations,
        }

        self._initialize_unique_labels()
        self._main()

        communities = self._get_communities()
        if len(communities) > 1:
            result = self._dfs_connectivity(communities)
        else:
            result = [1]
        if len(result) > len(communities):
            print("Disconnected communities found!")
            for index, community in enumerate(result):
                for member in community:
                    self._node_labels[member] = index

        return self._network, self._node_labels

    def _get_communities(self):
        unique_community_labels = set(self._node_labels.values())
        unique_communities_by_label = {value: [] for value in unique_community_labels}
        for node, label in self._node_labels.items():
            unique_communities_by_label[label].append(node)
        return unique_communities_by_label.values()

    def _initialize_unique_labels(self):
        for label, node in enumerate(self._network.nodes()):
            self._node_labels[node] = label

    def _inclusion(self, node, label_cnt_dict, max_labels, max_label_value):
        if self._node_labels[node] in max_labels:
            return self._node_labels[node]
        elif self._node_labels[node] in label_cnt_dict and max_label_value - label_cnt_dict[
          self._node_labels[node]] == 1:
            max_labels.append(self._node_labels[node])
            return random.choice(max_labels)
        return random.choice(max_labels)

    def _retention(self, node, max_labels):
        if self._node_labels[node] in max_labels:
            return self._node_labels[node]
        return random.choice(max_labels)

    def _maximal_neighbouring_label(self, node):
        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter. Choose between \"random\", \"inclusion\" and \"retention\".")

        neighbor_labels = [self._node_labels[neighbor_node] for neighbor_node in self._network[node].keys()]
        unique_neighbor_labels, unique_neighbor_label_counts = np.unique(np.array(neighbor_labels), return_counts=True)
        neighbor_label_counts_dict = dict(zip(unique_neighbor_labels, unique_neighbor_label_counts))
        maximal_label_count = np.max(unique_neighbor_label_counts)
        maximal_labels = unique_neighbor_labels[
            np.argwhere(unique_neighbor_label_counts == np.amax(unique_neighbor_label_counts)).flatten()].tolist()

        """
        labels = [self._label_map[adj] for adj in self.graph[node].keys()]
        label_cnt_dict = Counter(labels)
        max_label_value = max(label_cnt_dict.values())
        max_label_cnt = {key: max_label_value for key in label_cnt_dict.keys() if
                         label_cnt_dict[key] == max_label_value}
        max_labels = list(max_label_cnt.keys())
        """

        if len(maximal_labels) == 1:
            return maximal_labels[0]
        elif self._settings["label_ties_resolution"] == "random":
            return random.choice(maximal_labels)
        elif self._settings["label_ties_resolution"] == "inclusion":
            return self._inclusion(node, neighbor_label_counts_dict, maximal_labels, maximal_label_count)
        elif self._settings["label_ties_resolution"] == "retention":
            return self._retention(node, maximal_labels)

    def _maximal_neighbouring_weight(self, node):
        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter. Choose between \"random\", \"inclusion\" and \"retention\".")

        weights = {}
        for neighbor_node in self._network[node].keys():
            if self._node_labels[neighbor_node] in weights:
                weights[self._node_labels[neighbor_node]] += self._network[node][neighbor_node]["weight"]
            else:
                weights[self._node_labels[neighbor_node]] = self._network[node][neighbor_node]["weight"]
        max_weight_value = max(weights.values())
        max_weight_labels = []
        for neighbor_label in weights.keys():
            if weights[neighbor_label] == max_weight_value:
                max_weight_labels.append(neighbor_label)

        if len(max_weight_labels) == 1:
            return max_weight_labels[0]
        elif self._settings["label_ties_resolution"] in ["random", "inclusion"]:
            return random.choice(max_weight_labels)
        elif self._settings["label_ties_resolution"] == "retention":
            return self._retention(node, max_weight_labels)

    # A function used by DFS
    def _dfs_recursive(self, v, visited, community, connected_group):
        visited[v] = True

        for i in self._network[v].keys():
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
        if self._settings["convergence_criterium"] not in ["label-equilibrium", "strong-community"]:
            raise ValueError("Invalid label equilibrium criteria parameter. Choose between \"label-equilibrium\", \"strong-community\" and \"change\".")

        for node in self._network.nodes():
            labels = [self._node_labels[adj] for adj in self._network[node].keys()]
            unique, counts = np.unique(np.array(labels), return_counts=True)
            max_labels = unique[np.argwhere(counts == np.amax(counts)).flatten()].tolist()

            if len(max_labels) > 1:
                if self._settings["convergence_criterium"] == "label-equilibrium":
                    continue
                elif self._settings["convergence_criterium"] == "strong-community":
                    return True
            elif self._node_labels[node] != max_labels[0]:
                return True
        return False

    def _asynchronous_propagation(self):
        change = False
        for node in random.sample(self._network.nodes(), len(self._network.nodes())):
            if self._settings["weighted"]:
                new_label = self._maximal_neighbouring_weight(node)
            else:
                new_label = self._maximal_neighbouring_label(node)
            if self._node_labels[node] != new_label:
                self._node_labels[node] = new_label
                change = True
        return change

    def _synchronous_propagation(self):
        change = False
        sync_label_map = copy.deepcopy(self._node_labels)
        for node in random.sample(self._network.nodes(), len(self._network.nodes())):
            if self._settings["weighted"]:
                new_label = self._maximal_neighbouring_weight(node)
            else:
                new_label = self._maximal_neighbouring_label(node)
            if sync_label_map[node] != new_label:
                sync_label_map[node] = new_label
                change = True
        self._node_labels = sync_label_map
        return change

    def _main(self):
        change = True
        self.iterations = 0
        while change and self.iterations < self._settings["maximum_iterations"]:
            self.iterations += 1
            if self._settings["order_of_label_propagation"] == "synchronous":
                change = self._synchronous_propagation()
            elif self._settings["order_of_label_propagation"] == "asynchronous":
                change = self._asynchronous_propagation()
            else:
                raise ValueError("Invalid iteration order parameter")
            if self._settings["convergence_criterium"] != "change":
                change = self._convergence()

    def consensus_clustering(self, label_ties_resolution, convergence_criterium, order, threshold, number_of_partitions, max_recursive_steps=10, weighted=False, maximum_iterations=100, fcc=False, convergence_factor=0.02):
        if weighted and not self._network_type == "W":
            raise ValueError("Cannot perform label propagation that includes weighted edges, because graph type is not \"W\" (Weighted)")
        if weighted and label_ties_resolution == "inclusion":
            warnings.warn("Inclusion cannot be used on weighted graphs, random resolution will be performed instead")
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

        found_communities = {}
        for i in range(self._settings["number_of_partitions"]):
            self._initialize_unique_labels()
            self._main()
            found_communities[i] = self._node_labels

        self._settings["weighted"] = True
        self.recursive_steps = 0
        self._recursive(found_communities)
        return self._network, self._node_labels

    def _recursive(self, found_communities):
        if self.recursive_steps >= self._settings["max_recursive_steps"]:
            print("Reached maximum recursive steps")
            return

        for u, v in self._network.edges():
            self._network[u][v]['weight'] = 0.0

        for node, nbr in self._network.edges():
            for i in range(self._settings["number_of_partitions"]):
                communities = found_communities[i]
                if communities[node] == communities[nbr]:
                    if not self._network.has_edge(node, nbr):
                        self._network.add_edge(node, nbr, weight=0)
                    self._network[node][nbr]['weight'] += 1

        edges_to_be_removed = []
        for u, v in self._network.edges():
            if self._network[u][v]['weight'] < self._settings["threshold"] * self._settings["number_of_partitions"]:
                edges_to_be_removed.append((u, v))

        self._network.remove_edges_from(edges_to_be_removed)

        if self._settings["fcc"]:
            for _ in range(self._network.number_of_edges()):
                node = np.random.choice(self._network.nodes())
                neighbors = [n[1] for n in self._network.edges(node)]

                if len(neighbors) >= 2:
                    j, k = random.sample(set(neighbors), 2)

                    if not self._network.has_edge(j, k):
                        self._network.add_edge(j, k, weight=0)

                        for i in range(self._settings["number_of_partitions"]):
                            communities = found_communities[i]
                            if communities[j] == communities[k]:
                                self._network[j][k]['weight'] += 1

        if self._is_block_diagonal():
            return

        found_communities = {}
        for i in range(self._settings["number_of_partitions"]):
            self._initialize_unique_labels()
            self._main()
            found_communities[i] = self._node_labels

        self.recursive_steps += 1
        self._recursive(found_communities)
        return

    def _is_block_diagonal(self):
        count_faulty_edges = 0
        for weight in nx.get_edge_attributes(self._network, 'weight').values():
            if self._settings["threshold"] < weight < self._settings["number_of_partitions"]:
                if self._settings["fcc"]:
                    count_faulty_edges += 1
                else:
                    return False

        if count_faulty_edges > self._settings["convergence_factor"] * self._network.number_of_edges():
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
