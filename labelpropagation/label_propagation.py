import networkx as nx, matplotlib.pyplot as plt, random, operator, time
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self._G = nx.Graph()
        # TODO: Adjust read file to handle weighted graphs
        edges = read_file(file_path)
        self._G.add_edges_from(edges)
        self._label_map = {}
        self._iterations = 0

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
        self._iterations = 0

    def evaluate(self, label_ties_resolution, label_equilibrium_criteria,
                 order_of_label_propagation, k, maximum_iterations=100):
        """
        Runs the algorithm k times, and prints the average number of communities found.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run100("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        final_number_of_groups = []
        runtime_list = []
        iteration_list = []
        for i in range(k):
            self._initialize_labels()
            start_time = time.clock()
            self._algorithm(label_ties_resolution, label_equilibrium_criteria,
                            order_of_label_propagation, maximum_iterations)
            runtime_list.append(time.clock() - start_time)
            iteration_list.append(self._iterations)
            self._iterations = 0
            final_number_of_groups.append(self._get_unique_groups())
        self._print_results_of_evaluate(k, final_number_of_groups, runtime_list, iteration_list)

    def _print_results_of_run(self, runtime):
        print("-Run method-")
        print("Number of communities found: %s" % self._get_unique_groups())
        print("Number of iterations: %d" % self._iterations)
        print("Time elapsed: %f milliseconds" % (runtime * 1000))
        print()

    # TODO: Think of a more clever way to write static methods
    def _print_results_of_evaluate(self, k, final_number_of_groups, runtime_list, iteration_list):
        print("-Evaluate method-")
        print("Average number of communities found in %d attempts: %s" % (k, average(final_number_of_groups)))
        counted_communities = Counter(final_number_of_groups)
        for key in counted_communities.keys():
            print("\tIn %d attempts number of communities found was %d" % (counted_communities[key], key))

        print("Average number of iterations in %d attempts: %s" % (k, average(iteration_list)))
        counted_iterations = Counter(iteration_list)
        for key in counted_iterations.keys():
            print("\tIn %d attempts number of iterations was %d" % (counted_iterations[key], key))

        print("Average time elapsed in %d attempts: %f milliseconds" % (k, float(average(runtime_list)) * 1000))
        print()

    def _get_unique_groups(self):
        return len(Counter(self._label_map.values()))

    def _initialize_labels(self):
        for i, node in enumerate(self._G.nodes()):
            self._label_map[node] = i

    def _draw_graph(self):
        colors = [self._label_map.get(node) for node in self._G.nodes()]
        plt.subplot(111)
        nx.draw(self._G, with_labels=self._G.nodes, node_color=colors)
        plt.show()

    def _max_neighbouring_label(self, node, label_ties_resolution):
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

    def _iteration_order(self, order):
        if order == "synchronous":
            return self._G.nodes()
        elif order == "asynchronous":
            return random.sample(self._G.nodes(), len(self._G.nodes()))

    def _algorithm(self, label_ties_resolution, label_equilibrium_criteria,
                   order_of_label_propagation, maximum_iterations):
        change = True
        while change and self._iterations < maximum_iterations:
            self._iterations = self._iterations + 1
            change = False
            for node in self._iteration_order(order_of_label_propagation):
                new_label = self._max_neighbouring_label(node, label_ties_resolution)
                if self._label_map[node] != new_label:
                    self._label_map[node] = new_label
                    change = True
            if label_equilibrium_criteria != "change":
                change = self._convergence(label_equilibrium_criteria)


def argument_check(label_ties_resolution, label_equilibrium_criteria, order_of_label_propagation):
    if label_ties_resolution not in ["random", "inclusion", "retention"]:
        raise ValueError("Invalid label ties resolution parameter")
    if label_equilibrium_criteria not in ["change", "label-equilibrium", "strong-community"]:
        raise ValueError("Invalid label equilibrium criteria parameter")
    if order_of_label_propagation not in ["synchronous", "asynchronous"]:
        raise ValueError("Invalid iteration order parameter")


def average(lst):
    return sum(lst) / len(lst)


def all_labels_maximal(labels):
    label_count = list(Counter(labels).values())
    if len(label_count) == 1:
        return False
    for label_cnt in label_count:
        if label_count[0] != label_cnt:
            return False
    return True


def read_file(file_path):
    with open(file_path, "rt") as f:
        edges = []
        for line in f:
            line = line.split()
            if line[0].isalnum():
                edges.append((line[0], line[1]))
            else:
                continue
        return edges
