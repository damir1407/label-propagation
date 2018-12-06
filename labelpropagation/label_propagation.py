from collections import Counter
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import copy
import warnings


class LabelPropagation:
    def __init__(self, file_path, graph_type="U"):
        """
        Initialization of object attributes.
        """
        self._G = nx.Graph()
        self._graph_type = graph_type
        self._initialize_graph_from_file(file_path)
        self._orig_G = self._G
        self._label_map = {}
        self._iterations = 0
        self._final_number_of_groups = []
        self._runtime_list = []
        self._iteration_list = []
        self._settings = {}

    def _initialize_graph_from_file(self, file_path):
        """
        Creates the graph from input-examples file, based on whether it's weighted or unweighted.
        """
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
                self._G.add_edges_from(edges)
            elif self._graph_type == "W":
                self._G.add_weighted_edges_from(edges)

    def run(self, label_resolution, equilibrium, order,
            number_of_repetitions=1, include_weights=False, draw=False, maximum_iterations=100):
        """
        Runs the algorithm once, and presents a drawing of the result.
        Usage:
            lp = LabelPropagation("path/to/input-examples/file")
            lp.run(label_resolution="retention", equilibrium="change", order="asynchronous", draw=True, include_weights=True)
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        if include_weights is True and not self._graph_type == "W":
            raise ValueError("Cannot perform label propagation that includes weighted edges, "
                             "because graph type is not \"W\" (Weighted)")
        if include_weights is True and label_resolution == "inclusion":
            warnings.warn("Inclusion cannot be used on weighted graphs, random resolution will be performed instead")

        self._settings = {
            "label_ties_resolution": label_resolution,
            "label_equilibrium_criteria": equilibrium,
            "order_of_label_propagation": order,
            "number_of_repetitions": number_of_repetitions,
            "include_weights": include_weights,
            "draw": draw,
            "maximum_iterations": maximum_iterations,
        }

        for i in range(self._settings["number_of_repetitions"]):
            self._initialize_labels()
            self._draw_graph()

            start_time = time.clock()
            self._algorithm()
            self._runtime_list.append(time.clock() - start_time)
            self._draw_graph()

            self._iteration_list.append(self._iterations)
            self._iterations = 0

            self._final_number_of_groups.append(self._get_number_of_communities())
        if self._settings["number_of_repetitions"] == 1:
            self._print_results_of_run()
        else:
            self._print_results_of_evaluate()
        self._reinitialise_attributes()

    # TODO: Double check and test consensus clustering; Write down open questions
    def consensus_clustering(self, label_resolution, equilibrium, order, threshold, draw=False,
                             maximum_iterations=100):
        """
        Consensus clustering algorithm
        """
        if not self._graph_type == "W":
            raise ValueError("Cannot perform label propagation that includes weighted edges, "
                             "because graph type is not \"W\" (Weighted)")
        if label_resolution == "inclusion":
            warnings.warn("Inclusion cannot be used on weighted graphs, random resolution will be performed instead")

        self._settings = {
            "label_ties_resolution": label_resolution,
            "label_equilibrium_criteria": equilibrium,
            "order_of_label_propagation": order,
            "threshold": threshold,
            "draw": draw,
            "maximum_iterations": maximum_iterations,
            "include_weights": True,
            "number_of_repetitions": 1,
            "number_of_nodes": len(self._G),
        }

        start_time = time.clock()

        # STEP 1 - Apply algorithm on G nP times, so to yield nP partitions.
        all_communities = []
        for i in range(self._settings["number_of_nodes"]):
            self._initialize_labels()
            self._algorithm()
            all_communities.append(self._get_communities())
            self._iterations = 0
        self._reinitialise_attributes()

        d_matrix = None
        new_d_matrix = None

        while True:
            # STEP 2 - Compute the consensus matrix D, where Dij is the number of partitions in which
            # vertices i and j of G are assigned to the same cluster, divided by nP.
            if d_matrix:
                d_matrix = self._reset_d_matrix(d_matrix)
            else:
                d_matrix = self._get_clean_d_matrix(self._G.__dict__["_adj"])
            for community in all_communities:
                for comm in community:
                    for k1, k2 in combinations(comm, r=2):
                        if k2 not in d_matrix[k1]:
                            d_matrix[k1][k2] = {"weight": 1}
                        d_matrix[k1][k2]["weight"] += 1

            # STEP 3 -  All entries of D below a chosen threshold τ are set to zero. TODO: Define threshold
            for k1, k2 in combinations(d_matrix.keys(), r=2):
                if k2 in d_matrix[k1] and d_matrix[k1][k2]["weight"] != 0:
                    # divide by nP
                    d_ij = d_matrix[k1][k2]["weight"] / self._settings["number_of_nodes"]
                    if d_ij < self._settings["threshold"]:
                        d_matrix[k1][k2]["weight"] = 0
                    else:
                        d_matrix[k1][k2]["weight"] = d_ij

            # STEP 4 -  Apply algorithm on D nP times, so to yield nP partitions.
            self._G = nx.Graph(d_matrix)
            all_communities = []
            for i in range(self._settings["number_of_nodes"]):
                self._initialize_labels()
                self._algorithm()
                all_communities.append(self._get_communities())
                self._iterations = 0
            self._reinitialise_attributes()

            # STEP 4.1
            if new_d_matrix:
                new_d_matrix = self._reset_d_matrix(new_d_matrix)
            else:
                new_d_matrix = self._get_clean_d_matrix(d_matrix)
            for community in all_communities:
                for comm in community:
                    for k1, k2 in combinations(comm, r=2):
                        if k2 not in new_d_matrix[k1]:
                            new_d_matrix[k1][k2] = {"weight": 1}
                        new_d_matrix[k1][k2]["weight"] += 1

            # STEP 5 - If the partitions are all equal, stop (the consensus matrix would be block-diagonal).
            # Otherwise go back to 2.
            if self._is_block_diagonal(new_d_matrix):
                self._G = nx.Graph(new_d_matrix)
                break
        self._runtime_list.append(time.clock() - start_time)
        self._iteration_list.append(0)
        self._final_number_of_groups.append(self._get_number_of_communities())
        self._print_results_of_run()
        self._draw_graph()

    def _print_results_of_run(self):
        """
        Print results of run function.
        """
        print("Number of communities found: %s" % self._final_number_of_groups[0])
        print("Number of iterations: %d" % self._iteration_list[0])
        print("Time elapsed: %f milliseconds" % (self._runtime_list[0] * 1000))
        print()

    def _print_results_of_evaluate(self):
        """
        Print results of evaluate function.
        """
        print("Average number of communities found in %d attempts: %s" % (
            self._settings["number_of_repetitions"], np.average(self._final_number_of_groups)))
        counted_communities = Counter(self._final_number_of_groups)
        for key in counted_communities.keys():
            print("\tIn %d attempts number of communities found was %d" % (counted_communities[key], key))

        print("Average number of iterations in %d attempts: %s" %
              (self._settings["number_of_repetitions"], np.average(self._iteration_list)))
        counted_iterations = Counter(self._iteration_list)
        for key in counted_iterations.keys():
            print("\tIn %d attempts number of iterations was %d" % (counted_iterations[key], key))

        print("Average time elapsed in %d attempts: %f milliseconds" %
              (self._settings["number_of_repetitions"], float(np.average(self._runtime_list)) * 1000))
        print()

    def _reinitialise_attributes(self):
        """
        Reinitialization of object attributes.
        """
        self._iterations = 0
        self._final_number_of_groups = []
        self._runtime_list = []
        self._iteration_list = []

    def _is_block_diagonal(self, new_d_matrix):
        for k1, k2 in combinations(new_d_matrix.keys(), r=2):
            if k2 in new_d_matrix[k1]:
                new_d_ij = new_d_matrix[k1][k2]["weight"] / self._settings["number_of_nodes"]
                if 0 < new_d_ij < 1:
                    return False
                else:
                    new_d_matrix[k1][k2]["weight"] = new_d_ij
        return True

    @staticmethod
    def _get_clean_d_matrix(input_dict):
        temp_dict = copy.deepcopy(input_dict)
        for key, val in temp_dict.items():
            for key2, val2 in val.items():
                val2["weight"] = 0
        return temp_dict

    @staticmethod
    def _reset_d_matrix(input_dict):
        for key, val in input_dict.items():
            for key2, val2 in val.items():
                val2["weight"] = 0
        return input_dict

    def _get_communities(self):
        community_set = set(self._label_map.values())
        community_dict = {value: [] for value in community_set}
        for key, value in self._label_map.items():
            community_dict[value].append(key)
        return community_dict.values()

    def _get_number_of_communities(self):
        """
        Returns number of communities found.
        """
        return len(Counter(self._label_map.values()))

    def _initialize_labels(self):
        """
        Initialization of graph labels.
        """
        for label, node in enumerate(self._G.nodes()):
            self._label_map[node] = label

    def _draw_graph(self):
        """
        Drawing the image of the graph.
        """
        if self._settings["draw"] and len(self._G.nodes()) < 50 and self._settings["number_of_repetitions"] == 1:
            colors = [self._label_map.get(node) for node in self._G.nodes()]
            plt.subplot(111)
            nx.draw(self._G, with_labels=self._G.nodes, node_color=colors)
            plt.show()

    def _maximal_neighbouring_label(self, node):
        """
        Algorithm help function, which finds the maximal neighbouring label based on the label_ties_resolution string.
        """
        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter")

        labels = [self._label_map[adj] for adj in self._G[node].keys()]
        label_count = Counter(labels)
        max_value = max(label_count.values())
        max_value_label_count = {key: max_value for key in label_count.keys() if label_count[key] == max_value}

        if len(max_value_label_count) == 1:
            return list(max_value_label_count.keys())[0]
        elif self._settings["label_ties_resolution"] == "random":
            return random.choice(list(max_value_label_count.keys()))
        elif self._settings["label_ties_resolution"] == "inclusion":
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
        elif self._settings["label_ties_resolution"] == "retention":
            if self._label_map[node] in max_value_label_count.keys():
                return self._label_map[node]
            else:
                return random.choice(list(max_value_label_count.keys()))

    def _maximal_neighbouring_weight(self, node):
        """
        Algorithm help function, which finds the maximal neighbouring label based on the neighbouring weights.
        """
        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter")

        weights = {self._label_map[adj]: 0 for adj in self._G[node].keys()}
        for adj in self._G[node].keys():
            weights[self._label_map[adj]] = weights[self._label_map[adj]] + self._G[node][adj]["weight"]
        max_value = max(weights.values())
        max_value_weight_count = {key: max_value for key in weights.keys() if weights[key] == max_value}

        if len(max_value_weight_count) == 1:
            return list(max_value_weight_count.keys())[0]
        elif self._settings["label_ties_resolution"] in ["random", "inclusion"]:
            return random.choice(list(max_value_weight_count.keys()))
        elif self._settings["label_ties_resolution"] == "retention":
            if self._label_map[node] in max_value_weight_count.keys():
                return self._label_map[node]
            else:
                return random.choice(list(max_value_weight_count.keys()))

    def _convergence(self):
        """
        Algorithm help function, which affects convergence based on label_equilibrium_criteria string.
        """
        if self._settings["label_equilibrium_criteria"] not in ["label-equilibrium", "strong-community"]:
            raise ValueError("Invalid label equilibrium criteria parameter")

        # TODO: Double check label-equilibrium criteria
        for node in self._G.nodes():
            labels = [self._label_map[adj] for adj in self._G[node].keys()]
            label_count = Counter(labels)
            max_value = max(label_count.values())
            max_value_label_count = {key: max_value for key in label_count.keys() if label_count[key] == max_value}

            if len(max_value_label_count) > 1:
                if self._settings["label_equilibrium_criteria"] == "label-equilibrium":
                    continue
                elif self._settings["label_equilibrium_criteria"] == "strong-community":
                    return True
            if self._label_map[node] != list(max_value_label_count.keys())[0]:
                return True
        return False

    def _asynchronous_propagation(self):
        change = False
        for node in random.sample(self._G.nodes(), len(self._G.nodes())):
            if self._settings["include_weights"]:
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
        for node in random.sample(self._G.nodes(), len(self._G.nodes())):
            if self._settings["include_weights"]:
                new_label = self._maximal_neighbouring_weight(node)
            else:
                new_label = self._maximal_neighbouring_label(node)
            if sync_label_map[node] != new_label:
                sync_label_map[node] = new_label
                change = True
        self._label_map = sync_label_map
        return change

    def _algorithm(self):
        """
        Main algorithm function.
        """
        change = True
        while change and self._iterations < self._settings["maximum_iterations"]:
            self._iterations = self._iterations + 1
            if self._settings["order_of_label_propagation"] == "synchronous":
                change = self._synchronous_propagation()
            elif self._settings["order_of_label_propagation"] == "asynchronous":
                change = self._asynchronous_propagation()
            else:
                raise ValueError("Invalid iteration order parameter")
            if self._settings["label_equilibrium_criteria"] != "change":
                change = self._convergence()
