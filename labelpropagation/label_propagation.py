import networkx as nx, matplotlib.pyplot as plt, random, operator
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self._G = nx.Graph()
        edges = read_file(file_path)
        self._G.add_edges_from(edges)
        self._label_map = {}
        self._final_groups = []

    def run(self, label_ties_resolution):
        """
        Runs the algorithm once, and presents a drawing of the result.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        self._initialize_labels()
        self._draw()
        self._algorithm(label_ties_resolution)
        self._draw()

    def run100(self, label_ties_resolution):
        """
        Runs the algorithm hundred times, and prints the average number of communities found.
        Usage:
            lp = LabelPropagation("path/to/input/file")
            lp.run100("label_ties_resolution_string")
        More details about "label_ties_resolution_string" can be found in the README file.
        """
        for i in range(100):
            self._initialize_labels()
            self._algorithm(label_ties_resolution)
            self._final_groups.append(self._get_unique_groups())
        self._print_results_of_run100()

    def _print_results_of_run100(self):
        print("Average number of communities found in 100 attempts: %s" % self._get_average_number_of_groups())
        counted = Counter(self._final_groups)
        for key in counted.keys():
            print("In %d attempts number of communities found was %d" % (counted[key], key))

    def _get_unique_groups(self):
        return Counter(self._label_map.values()).__len__()

    def _get_average_number_of_groups(self):
        return sum(self._final_groups) / len(self._final_groups)

    def _initialize_labels(self):
        for i, node in enumerate(self._G.nodes()):
            self._label_map[node] = i

    def _draw(self):
        colors = [self._label_map.get(node) for node in self._G.nodes()]
        nx.draw(self._G, with_labels=self._G.nodes, node_color=colors)
        plt.show()

    def _find_max_label(self, node, label_ties_resolution):
        labels = [self._label_map[adj] for adj in self._G[node].keys()]
        label_count = list(Counter(labels).values())
        if all(label_count[0] == label_cnt for label_cnt in label_count):
            if label_ties_resolution == "random":
                return random.choice(labels)
            elif label_ties_resolution == "leung":
                labels.append(self._label_map[node])
            elif label_ties_resolution == "barberclark":
                return self._label_map[node]
        return max(Counter(labels).items(), key=operator.itemgetter(1))[0]

    def _algorithm(self, label_ties_resolution):
        change = True
        while change:
            change = False
            for node in sorted(self._G.nodes()):
                new_label = self._find_max_label(node, label_ties_resolution)
                if self._label_map[node] != new_label:
                    self._label_map[node] = new_label
                    change = True


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
