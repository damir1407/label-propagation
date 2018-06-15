from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import time
import copy


class LabelPropagation:
    def __init__(self, file_path, graph_type="U"):
        """
        Initialization of object attributes.
        """
        self._G = nx.Graph()
        self.graph_type = graph_type
        self._create_graph_from_file(file_path)
        self._label_map = {}
        self._iterations = 0
        self._final_number_of_groups = []
        self._runtime_list = []
        self._iteration_list = []
        self._settings = {}

    def _create_graph_from_file(self, file_path):
        """
        Creates the graph from input file, based on whether it's weighted or unweighted.
        """
        with open(file_path, "rt") as f:
            edges = []
            for line in f:
                line = line.split()
                if self.graph_type == "U":
                    edges.append((line[0], line[1]))
                elif self.graph_type == "W":
                    edges.append((line[0], line[1], line[2]))
            if self.graph_type == "U":
                self._G.add_edges_from(edges)
            elif self.graph_type == "W":
                self._G.add_weighted_edges_from(edges)

    def run(self, maximal_label_condition, label_ties_resolution, label_equilibrium_criteria,
            order_of_label_propagation, draw=False, maximum_iterations=100):
        """
        Runs the algorithm once, and presents a drawing of the result.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        if maximal_label_condition == "weight" and not self.graph_type == "W":
            raise ValueError("Cannot perform weighted propagation, because graph type is not \"W\" (Weighted)")

        self._settings = {
            "maximal_label_condition": maximal_label_condition,
            "label_ties_resolution": label_ties_resolution,
            "label_equilibrium_criteria": label_equilibrium_criteria,
            "order_of_label_propagation": order_of_label_propagation,
            "maximum_iterations": maximum_iterations,
            "draw": draw,
        }

        self._initialize_labels()

        self._draw_graph()

        start_time = time.clock()
        self._algorithm()
        runtime = time.clock() - start_time

        self._draw_graph()
        self._print_results_of_run(runtime)
        self._reinitialise_attributes()

    def evaluate(self, maximal_label_condition, label_ties_resolution,
                 label_equilibrium_criteria, order_of_label_propagation, k, maximum_iterations=100):
        """
        Runs the algorithm k times, and prints the average number of communities found.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run100("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        if maximal_label_condition == "weight" and not self.graph_type == "W":
            raise ValueError("Cannot perform weighted propagation, because graph type is not \"W\" (Weighted)")

        self._settings = {
            "maximal_label_condition": maximal_label_condition,
            "label_ties_resolution": label_ties_resolution,
            "label_equilibrium_criteria": label_equilibrium_criteria,
            "order_of_label_propagation": order_of_label_propagation,
            "maximum_iterations": maximum_iterations,
        }

        for i in range(k):
            self._initialize_labels()

            start_time = time.clock()
            self._algorithm()
            self._runtime_list.append(time.clock() - start_time)

            self._iteration_list.append(self._iterations)
            self._iterations = 0

            self._final_number_of_groups.append(self._get_number_of_communities())
        self._print_results_of_evaluate(k)
        self._reinitialise_attributes()

    def _print_results_of_run(self, runtime):
        """
        Print results of run function.
        """
        print("-Run method-")
        print("Number of communities found: %s" % self._get_number_of_communities())
        print("Number of iterations: %d" % self._iterations)
        print("Time elapsed: %f milliseconds" % (runtime * 1000))
        print()

    def _print_results_of_evaluate(self, k):
        """
        Print results of evaluate function.
        """
        print("-Evaluate method-")
        print("Average number of communities found in %d attempts: %s" % (k, np.average(self._final_number_of_groups)))
        counted_communities = Counter(self._final_number_of_groups)
        for key in counted_communities.keys():
            print("\tIn %d attempts number of communities found was %d" % (counted_communities[key], key))

        print("Average number of iterations in %d attempts: %s" % (k, np.average(self._iteration_list)))
        counted_iterations = Counter(self._iteration_list)
        for key in counted_iterations.keys():
            print("\tIn %d attempts number of iterations was %d" % (counted_iterations[key], key))

        print("Average time elapsed in %d attempts: %f milliseconds" %
              (k, float(np.average(self._runtime_list)) * 1000))
        print()

    def _reinitialise_attributes(self):
        """
        Reinitialization of object attributes.
        """
        self._iterations = 0
        self._final_number_of_groups = []
        self._runtime_list = []
        self._iteration_list = []

    def _get_number_of_communities(self):
        """
        Returns number of communities found.
        """
        return len(Counter(self._label_map.values()))

    def _initialize_labels(self):
        """
        Initialization of graph labels.
        """
        for i, node in enumerate(self._G.nodes()):
            self._label_map[node] = i

    def _draw_graph(self):
        """
        Drawing the image of the graph.
        """
        if self._settings["draw"] and len(self._G.nodes()) < 50:
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
        label_count_dict = {key: max_value for key in label_count.keys() if label_count[key] == max_value}

        if len(label_count_dict) == 1:
            x = list(label_count_dict.keys())[0]
            return x
        else:
            if self._settings["label_ties_resolution"] == "random":
                return random.choice(list(label_count_dict.keys()))
            # TODO: Fix inclusion
            elif self._settings["label_ties_resolution"] == "inclusion":
                labels.append(self._label_map[node])
            elif self._settings["label_ties_resolution"] == "retention":
                if self._label_map[node] in label_count_dict.keys():
                    return self._label_map[node]
                else:
                    return random.choice(list(label_count_dict.keys()))

    # TODO: _maximal_neighbouring_weight method
    def _maximal_neighbouring_weight(self, node):

        if self._settings["label_ties_resolution"] not in ["random", "inclusion", "retention"]:
            raise ValueError("Invalid label ties resolution parameter")

        # weights = [self._G[node][adj]["weight"] for adj in self._G[node].keys()]
        # labels = [self._label_map[adj] for adj in self._G[node].keys()]
        weights = {self._label_map[adj]: 0 for adj in self._G[node].keys()}
        for adj in self._G[node].keys():
            weights[self._label_map[adj]] = weights[self._label_map[adj]] + int(self._G[node][adj]["weight"])
        print(weights)

    def _convergence(self):
        """
        Algorithm help function, which affects convergence based on label_equilibrium_criteria string.
        """
        if self._settings["label_equilibrium_criteria"] not in ["label-equilibrium", "strong-community"]:
            raise ValueError("Invalid label equilibrium criteria parameter")

        # TODO: Double check this
        for node in self._G.nodes():
            labels = [self._label_map[adj] for adj in self._G[node].keys()]
            label_count = Counter(labels)
            max_value = max(label_count.values())
            label_count_dict = {key: max_value for key in label_count.keys() if label_count[key] == max_value}

            if len(label_count_dict) > 1:
                if self._settings["label_equilibrium_criteria"] == "label-equilibrium":
                    continue
                elif self._settings["label_equilibrium_criteria"] == "strong-community":
                    return True
            if self._label_map[node] != list(label_count_dict.keys())[0]:
                return True
        return False

    def _asynchronous_propagation(self):
        change = False
        for node in random.sample(self._G.nodes(), len(self._G.nodes())):
            if self._settings["maximal_label_condition"] == "label":
                new_label = self._maximal_neighbouring_label(node)
            elif self._settings["maximal_label_condition"] == "weight":
                new_label = self._maximal_neighbouring_weight(node)
            else:
                raise ValueError("Invalid maximal label condition parameter")
            if self._label_map[node] != new_label:
                self._label_map[node] = new_label
                change = True
        return change

    def _synchronous_propagation(self):
        change = False
        sync_label_map = copy.deepcopy(self._label_map)
        for node in random.sample(self._G.nodes(), len(self._G.nodes())):
            if self._settings["maximal_label_condition"] == "label":
                new_label = self._maximal_neighbouring_label(node)
            elif self._settings["maximal_label_condition"] == "weight":
                new_label = self._maximal_neighbouring_weight(node)
            else:
                raise ValueError("Invalid maximal label condition parameter")
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
