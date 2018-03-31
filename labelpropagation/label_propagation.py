import networkx as nx, matplotlib.pyplot as plt, random, operator, time
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self._G = nx.Graph()
        edges = read_file(file_path)
        self._G.add_edges_from(edges)
        self._label_map = {}
        self._iterations = 0

    def run(self, label_ties_resolution, label_equilibrium_criterium, draw=False):
        """
        Runs the algorithm once, and presents a drawing of the result.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        self._initialize_labels()
        if draw:
            self._draw_graph()
        start_time = time.clock()
        self._algorithm(label_ties_resolution, label_equilibrium_criterium)
        runtime = time.clock() - start_time
        if draw:
            self._draw_graph()
        self._print_results_of_run(runtime)

    def evaluate(self, label_ties_resolution, label_equilibrium_criterium, k):
        """
        Runs the algorithm k times, and prints the average number of communities found.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run100("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        final_number_of_groups = []
        runtimes = []
        for i in range(k):
            self._initialize_labels()
            start_time = time.clock()
            self._algorithm(label_ties_resolution, label_equilibrium_criterium)
            runtimes.append(time.clock() - start_time)
            final_number_of_groups.append(self._get_unique_groups())
        self._print_results_of_evaluate(k, final_number_of_groups, runtimes)

    def _print_results_of_run(self, runtime):
        print("-Run method-")
        print("Number of communities found: %s" % self._get_unique_groups())
        print("Time elapsed: %f miliseconds" % (runtime * 1000))
        print("Number of iterations: %d" % self._iterations)
        print()

    def _print_results_of_evaluate(self, k, final_number_of_groups, runtimes):
        print("-Evaluate method-")
        print("Average number of communities found in %d attempts: %s" % (k, sum(final_number_of_groups) / len(final_number_of_groups)))
        print("Average time elapsed in %d attempts: %f miliseconds" % (k, float(sum(runtimes) / len(runtimes)) * 1000))
        counted = Counter(final_number_of_groups)
        for key in counted.keys():
            print("In %d attempts number of communities found was %d" % (counted[key], key))
        print()

    def _get_unique_groups(self):
        return Counter(self._label_map.values()).__len__()

    def _initialize_labels(self):
        for i, node in enumerate(self._G.nodes()):
            self._label_map[node] = i

    def _draw_graph(self):
        colors = [self._label_map.get(node) for node in self._G.nodes()]
        plt.subplot(111)
        nx.draw(self._G, with_labels=self._G.nodes, node_color=colors)
        plt.show()

    def _find_max_label(self, node, label_ties_resolution):
        labels = [self._label_map[adj] for adj in self._G[node].keys()]
        label_count = list(Counter(labels).values())
        if all(label_count[0] == label_cnt for label_cnt in label_count):
            if label_ties_resolution == "random":
                return random.choice(labels)
            elif label_ties_resolution == "inclusion":
                labels.append(self._label_map[node])
            elif label_ties_resolution == "retention":
                #TODO: Fix label retention
                return self._label_map[node]
        return max(Counter(labels).items(), key=operator.itemgetter(1))[0]

    #TODO: Rename raghavan function
    def _raghavan(self):
        for node in self._G.nodes():
            labels = [self._label_map[adj] for adj in self._G[node].keys()]
            #TODO: Check if neighbouring labels are equal
            #label_count = list(Counter(labels).values())
            if self._label_map[node] != max(Counter(labels).items(), key=operator.itemgetter(1))[0]:
                return True
        return False

    def _algorithm(self, label_ties_resolution, label_equilibrium_criterium):
        change = True
        while change:
            self._iterations = self._iterations + 1
            change = False
            for node in random.sample(self._G.nodes(), len(self._G.nodes())):
                new_label = self._find_max_label(node, label_ties_resolution)
                if self._label_map[node] != new_label:
                    self._label_map[node] = new_label
                    change = True
            if label_equilibrium_criterium == "raghavan":
                change = self._raghavan()


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
