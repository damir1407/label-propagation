from collections import Counter
from itertools import combinations
import networkx as nx
import random
import copy
import warnings


class LabelPropagation:
    def __init__(self, file_path, graph_type="U"):
        self._graph_type = graph_type
        self.graph = self._init_graph_from_file(file_path)
        self._label_map = {}
        self._settings = {}
        self._recursive_steps = None
        self._init_labels()

    def _init_graph_from_file(self, file_path):
        new_graph = nx.Graph()
        with open(file_path, "rt") as f:
            edges = []
            for line in f:
                line = line.split()
                if self._graph_type == "W" and len(line) < 3:
                    raise ValueError("Input file does not contain weights.")
                if self._graph_type == "U":
                    edges.append((line[0], line[1]))
                elif self._graph_type == "W":
                    edges.append((line[0], line[1], int(line[2])))
            if self._graph_type == "U":
                new_graph.add_edges_from(edges)
            elif self._graph_type == "W":
                new_graph.add_weighted_edges_from(edges)
        return new_graph

    def run(self, label_resolution, equilibrium, order, weighted=False, maximum_iterations=100):
        if weighted and not self._graph_type == "W":
            raise ValueError("Cannot perform label propagation that includes weighted edges, "
                             "because graph type is not \"W\" (Weighted)")
        if weighted and label_resolution == "inclusion":
            warnings.warn("Inclusion cannot be used on weighted graphs, random resolution will be performed instead")

        self._settings = {
            "label_ties_resolution": label_resolution,
            "label_equilibrium_criteria": equilibrium,
            "order_of_label_propagation": order,
            "weighted": weighted,
            "maximum_iterations": maximum_iterations,
        }

        self._init_labels()
        self._algorithm()
        """
        communities = self._get_communities()
        for community in communities:
            print(community)
            for member in community:
                self.dfs(member)
                print()
        """
        return self._label_map

    def consensus_clustering(self, label_resolution, equilibrium, order, threshold, number_of_partitions,
                             recursive_steps=10, weighted=False, maximum_iterations=100):
        if weighted and not self._graph_type == "W":
            raise ValueError("Cannot perform label propagation that includes weighted edges, "
                             "because graph type is not \"W\" (Weighted)")
        if weighted and label_resolution == "inclusion":
            warnings.warn("Inclusion cannot be used on weighted graphs, random resolution will be performed instead")
        if threshold > number_of_partitions or threshold < 0:
            raise ValueError("Threshold must be between 0 and number of partitions")

        self._settings = {
            "label_ties_resolution": label_resolution,
            "label_equilibrium_criteria": equilibrium,
            "order_of_label_propagation": order,
            "threshold": threshold,
            "maximum_iterations": maximum_iterations,
            "weighted": weighted,
            "number_of_partitions": number_of_partitions,
            "recursive_steps": recursive_steps,
        }

        found_communities = []
        for i in range(self._settings["number_of_partitions"]):
            self._init_labels()
            self._algorithm()
            found_communities.append(self._get_communities())

        self._settings["weighted"] = True
        self._recursive_steps = 0
        self.recursive(self.graph.__dict__["_adj"], found_communities)
        return self._label_map

    def recursive(self, matrix, found_communities):
        if self._recursive_steps == self._settings["recursive_steps"]:
            print("Reached maximum recursive steps")
            return

        d_matrix = self._get_clean_d_matrix(matrix)
        d_matrix = self._compute_d_matrix(d_matrix, found_communities)

        for node1, node2 in combinations(d_matrix.keys(), r=2):
            if node2 in d_matrix[node1]:
                if d_matrix[node1][node2]["weight"] < self._settings["threshold"]:
                    del d_matrix[node1][node2]
                    del d_matrix[node2][node1]

        if self._is_block_diagonal(d_matrix):
            return

        self.graph = nx.Graph(d_matrix)
        found_communities = []
        for i in range(self._settings["number_of_partitions"]):
            self._init_labels()
            self._algorithm()
            found_communities.append(self._get_communities())

        self._recursive_steps += 1
        self.recursive(self.graph.__dict__["_adj"], found_communities)
        return

    def _is_block_diagonal(self, new_d_matrix):
        for k1, k2 in combinations(new_d_matrix.keys(), r=2):
            if k2 in new_d_matrix[k1]:
                if self._settings["threshold"] < new_d_matrix[k1][k2]["weight"] < self._settings["number_of_partitions"]:
                    return False
        return True

    @staticmethod
    def _get_clean_d_matrix(input_dict):
        for key, val in input_dict.items():
            for key2, val2 in val.items():
                val2["weight"] = 0
        return input_dict

    @staticmethod
    def _compute_d_matrix(d_matrix, found_communities):
        for community in found_communities:
            for members in community:
                for node1, node2 in combinations(members, r=2):
                    if node2 not in d_matrix[node1]:
                        d_matrix[node1][node2] = {"weight": 1}
                        d_matrix[node2][node1] = {"weight": 1}
                    else:
                        d_matrix[node1][node2]["weight"] += 1
        return d_matrix

    def _get_communities(self):
        community_set = set(self._label_map.values())
        community_dict = {value: [] for value in community_set}
        for key, value in self._label_map.items():
            community_dict[value].append(key)
        return community_dict.values()

    def _init_labels(self):
        for label, node in enumerate(self.graph.nodes()):
            self._label_map[node] = label

    def _inclusion(self, node, label_count, max_value_label_count, max_value):
        if self._label_map[node] in max_value_label_count.keys():
            return self._label_map[node]
        elif self._label_map[node] in label_count:
            if max_value - label_count[self._label_map[node]] > 1:
                return random.choice(list(max_value_label_count.keys()))
            else:
                max_value_label_count[self._label_map[node]] = max_value
                return random.choice(list(max_value_label_count.keys()))
        else:
            return random.choice(list(max_value_label_count.keys()))

    def _retention(self, node, max_value_count):
        if self._label_map[node] in max_value_count.keys():
            return self._label_map[node]
        else:
            return random.choice(list(max_value_count.keys()))

    def _maximal_neighbouring_label(self, node):
        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter")

        labels = [self._label_map[adj] for adj in self.graph[node].keys()]
        label_count = Counter(labels)
        max_value = max(label_count.values())
        max_value_label_count = {key: max_value for key in label_count.keys() if label_count[key] == max_value}

        if len(max_value_label_count) == 1:
            return list(max_value_label_count.keys())[0]
        elif self._settings["label_ties_resolution"] == "random":
            return random.choice(list(max_value_label_count.keys()))
        elif self._settings["label_ties_resolution"] == "inclusion":
            return self._inclusion(node, label_count, max_value_label_count, max_value)
        elif self._settings["label_ties_resolution"] == "retention":
            return self._retention(node, max_value_label_count)

    def _maximal_neighbouring_weight(self, node):
        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter")

        weights = {self._label_map[adj]: 0 for adj in self.graph[node].keys()}
        for adj in self.graph[node].keys():
            weights[self._label_map[adj]] = weights[self._label_map[adj]] + self.graph[node][adj]["weight"]
        max_value = max(weights.values())
        max_value_weight_count = {key: max_value for key in weights.keys() if weights[key] == max_value}

        if len(max_value_weight_count) == 1:
            return list(max_value_weight_count.keys())[0]
        elif self._settings["label_ties_resolution"] in ["random", "inclusion"]:
            return random.choice(list(max_value_weight_count.keys()))
        elif self._settings["label_ties_resolution"] == "retention":
            return self._retention(node, max_value_weight_count)

    # A function used by DFS
    def dfs_recursive(self, v, visited, community, rez):

        visited[v] = True

        for i in self.graph[v].keys():
            if i in community and not visited[i]:
                community.remove(i)
                rez.append(i)
                rez = self.dfs_recursive(i, visited, community, rez)
        return rez

    def dfs(self, communities):
        result = []

        for community in communities:
            temp_community = copy.deepcopy(community)
            # Mark all the vertices as not visited
            visited = {member: False for member in community}
            while len(temp_community) != 0:

                v = random.choice(temp_community)
                temp_community.remove(v)

                result.append(self.dfs_recursive(v, visited, temp_community, [v]))
        return result

    def _convergence(self):
        if self._settings["label_equilibrium_criteria"] not in ["label-equilibrium", "strong-community"]:
            raise ValueError("Invalid label equilibrium criteria parameter")

        for node in self.graph.nodes():
            labels = [self._label_map[adj] for adj in self.graph[node].keys()]
            label_count = Counter(labels)
            max_value = max(label_count.values())
            max_value_label_count = {key: max_value for key in label_count.keys() if label_count[key] == max_value}

            if len(max_value_label_count) > 1:
                if self._settings["label_equilibrium_criteria"] == "label-equilibrium":
                    continue
                elif self._settings["label_equilibrium_criteria"] == "strong-community":
                    return True
            elif self._label_map[node] != list(max_value_label_count.keys())[0]:
                return True
        return False

    def _asynchronous_propagation(self):
        change = False
        for node in random.sample(self.graph.nodes(), len(self.graph.nodes())):
            if self._settings["weighted"]:
                new_label = self._maximal_neighbouring_weight(node)
            else:
                new_label = self._maximal_neighbouring_label(node)
            if self._label_map[node] != new_label:
                self._label_map[node] = new_label
                change = True
        return change

    def _synchronous_propagation(self):
        change = False
        sync_label_map = copy.deepcopy(self._label_map)
        for node in random.sample(self.graph.nodes(), len(self.graph.nodes())):
            if self._settings["weighted"]:
                new_label = self._maximal_neighbouring_weight(node)
            else:
                new_label = self._maximal_neighbouring_label(node)
            if sync_label_map[node] != new_label:
                sync_label_map[node] = new_label
                change = True
        self._label_map = sync_label_map
        return change

    def _algorithm(self):
        change = True
        iterations = 0
        while change and iterations < self._settings["maximum_iterations"]:
            iterations = iterations + 1
            if self._settings["order_of_label_propagation"] == "synchronous":
                change = self._synchronous_propagation()
            elif self._settings["order_of_label_propagation"] == "asynchronous":
                change = self._asynchronous_propagation()
            else:
                raise ValueError("Invalid iteration order parameter")
            if self._settings["label_equilibrium_criteria"] != "change":
                change = self._convergence()
