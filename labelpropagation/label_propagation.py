import networkx as nx, matplotlib.pyplot as plt, random, operator, time, copy
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path, graph_type="U"):
        """
        Initialization of object attributes.
        """
        self._G = nx.Graph()
        self._create_graph_from_file(file_path, graph_type)
        self._label_map = {}
        self._iterations = 0
        self._final_number_of_groups = []
        self._runtime_list = []
        self._iteration_list = []

    def _create_graph_from_file(self, file_path, graph_type):
        """
        Creates the graph from input file, based on whether it's weighted or unweighted.
        """
        with open(file_path, "rt") as f:
            edges = []
            for line in f:
                line = line.split()
                if graph_type == "U":
                    edges.append((line[0], line[1]))
                elif graph_type == "W":
                    edges.append((line[0], line[1], line[2]))
            if graph_type == "U":
                self._G.add_edges_from(edges)
            elif graph_type == "W":
                self._G.add_weighted_edges_from(edges)

    def run(self, label_ties_resolution, label_equilibrium_criteria, order_of_label_propagation,
            draw=False, maximum_iterations=100):
        """
        Runs the algorithm once, and presents a drawing of the result.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        argument_check(label_ties_resolution, label_equilibrium_criteria, order_of_label_propagation)
        self._initialize_labels()
        if draw and len(self._G.nodes()) < 50:
            self._draw_graph()

        start_time = time.clock()
        self._algorithm(label_ties_resolution, label_equilibrium_criteria,
                        order_of_label_propagation, maximum_iterations)
        runtime = time.clock() - start_time

        if draw and len(self._G.nodes()) < 50:
            self._draw_graph()
        self._print_results_of_run(runtime)
        self._reinitialise_attributes()

    def evaluate(self, label_ties_resolution, label_equilibrium_criteria,
                 order_of_label_propagation, k, maximum_iterations=100):
        """
        Runs the algorithm k times, and prints the average number of communities found.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run100("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        argument_check(label_ties_resolution, label_equilibrium_criteria, order_of_label_propagation)

        for i in range(k):
            self._initialize_labels()

            start_time = time.clock()
            self._algorithm(label_ties_resolution, label_equilibrium_criteria,
                            order_of_label_propagation, maximum_iterations)
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
        print("Average number of communities found in %d attempts: %s" % (k, average(self._final_number_of_groups)))
        counted_communities = Counter(self._final_number_of_groups)
        for key in counted_communities.keys():
            print("\tIn %d attempts number of communities found was %d" % (counted_communities[key], key))

        print("Average number of iterations in %d attempts: %s" % (k, average(self._iteration_list)))
        counted_iterations = Counter(self._iteration_list)
        for key in counted_iterations.keys():
            print("\tIn %d attempts number of iterations was %d" % (counted_iterations[key], key))

        print("Average time elapsed in %d attempts: %f milliseconds" % (k, float(average(self._runtime_list)) * 1000))
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
        colors = [self._label_map.get(node) for node in self._G.nodes()]
        plt.subplot(111)
        nx.draw(self._G, with_labels=self._G.nodes, node_color=colors)
        plt.show()

    def _max_neighbouring_label(self, node, label_ties_resolution):
        """
        Algorithm help function, which finds the maximal neighbouring label based on the label_ties_resolution string.
        """
        labels = [self._label_map[adj] for adj in self._G[node].keys()]
        if all_labels_maximal(labels):
            if label_ties_resolution == "random":
                return random.choice(labels)
            elif label_ties_resolution == "inclusion":
                labels.append(self._label_map[node])
            elif label_ties_resolution == "retention":
                if self._label_map[node] in labels:
                    return self._label_map[node]
                else:
                    return random.choice(labels)
        return max(Counter(labels).items(), key=operator.itemgetter(1))[0]

    def _convergence(self, label_equilibrium_criteria):
        """
        Algorithm help function, which affects convergence based on label_equilibrium_criteria string.
        """
        for node in self._G.nodes():
            labels = [self._label_map[adj] for adj in self._G[node].keys()]
            if all_labels_maximal(labels):
                if label_equilibrium_criteria == "label-equilibrium":
                    continue
                elif label_equilibrium_criteria == "strong-community":
                    return True
            if self._label_map[node] != max(Counter(labels).items(), key=operator.itemgetter(1))[0]:
                return True
        return False

    def _asynchronous_propagation(self, label_ties_resolution):
        change = False
        for node in random.sample(self._G.nodes(), len(self._G.nodes())):
            new_label = self._max_neighbouring_label(node, label_ties_resolution)
            if self._label_map[node] != new_label:
                self._label_map[node] = new_label
                change = True
        return change

    # TODO: Double check synchronous propagation
    def _synchronous_propagation(self, label_ties_resolution):
        change = False
        sync_label_map = copy.deepcopy(self._label_map)
        for node in random.sample(self._G.nodes(), len(self._G.nodes())):
            new_label = self._max_neighbouring_label(node, label_ties_resolution)
            if sync_label_map[node] != new_label:
                sync_label_map[node] = new_label
                change = True
        self._label_map.clear()
        self._label_map.update(sync_label_map)
        sync_label_map.clear()
        return change

    def _algorithm(self, label_ties_resolution, label_equilibrium_criteria,
                   order_of_label_propagation, maximum_iterations):
        """
        Main algorithm function.
        """
        change = True
        while change and self._iterations < maximum_iterations:
            self._iterations = self._iterations + 1
            if order_of_label_propagation == "synchronous":
                change = self._synchronous_propagation(label_ties_resolution)
            elif order_of_label_propagation == "asynchronous":
                change = self._asynchronous_propagation(label_ties_resolution)
            if label_equilibrium_criteria != "change":
                change = self._convergence(label_equilibrium_criteria)


def argument_check(label_ties_resolution, label_equilibrium_criteria, order_of_label_propagation):
    """
    Help function which checks if user input arguments are valid.
    """
    if label_ties_resolution not in ["random", "inclusion", "retention"]:
        raise ValueError("Invalid label ties resolution parameter")
    if label_equilibrium_criteria not in ["change", "label-equilibrium", "strong-community"]:
        raise ValueError("Invalid label equilibrium criteria parameter")
    if order_of_label_propagation not in ["synchronous", "asynchronous"]:
        raise ValueError("Invalid iteration order parameter")


def average(input_list):
    """
    Help function, which returns the average of the given list.
    """
    return sum(input_list) / len(input_list)


def all_labels_maximal(labels):
    """
    Help function, which returns true if all neighbouring labels are maximal.
    """
    label_count = list(Counter(labels).values())
    if len(label_count) == 1:
        return False
    for label_cnt in label_count:
        if label_count[0] != label_cnt:
            return False
    return True
