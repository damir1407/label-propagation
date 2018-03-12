import networkx as nx, matplotlib.pyplot as plt, random, operator
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self._G = nx.Graph()
        self._G.add_edges_from(read_file(file_path))
        self._label_map = {}
        self._final_groups = []

    def run(self):
        self._initialize_labels()
        self._algorithm(True)

    def run100(self):
        for i in range(100):
            self._initialize_labels()
            self._algorithm(False)
            self._final_groups.append(self._get_unique_groups())
        self._print_results_of_run100()

    def _print_results_of_run100(self):
        print("Average number of communities found in 100 iterations: %s" % self._get_average_number_of_groups())
        counted = Counter(self._final_groups)
        for key in counted.keys():
            print("In %d iterations number of communities found was %d" % (counted[key], key))

    def _get_unique_groups(self):
        return Counter(self._label_map.values()).__len__()

    def _get_average_number_of_groups(self):
        return sum(self._final_groups) / len(self._final_groups)

    def _initialize_labels(self):
        for i, node in enumerate(self._G.nodes()):
            self._label_map[node] = i+1

    def _draw(self):
        values = [self._label_map.get(node) for node in self._G.nodes()]
        plt.subplot(111)
        nx.draw(self._G, with_labels=self._G.nodes, node_color=values)
        plt.show()

    def _algorithm(self, draw):
        if draw:
            self._draw()
        change = True
        while change:
            change = False
            for node in sorted(self._G.nodes()):
                labels = [self._label_map[adj] for adj in self._G[node].keys()]
                new_label = find_max_label(labels)
                if self._label_map[node] != new_label:
                    self._label_map[node] = new_label
                    change = True
        if draw:
            self._draw()


def find_max_label(labels):
    label_count = list(Counter(labels).values())
    if all(label_count[0] == label_cnt for label_cnt in label_count):
        return random.choice(labels)
    else:
        return max(Counter(labels).items(), key=operator.itemgetter(1))[0]

def read_file(file_path):
    with open(file_path, "rt") as f:
        data = []
        for line in f:
            line = line.split()
            data.append((line[0], line[1]))
        return data
