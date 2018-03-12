import networkx as nx, matplotlib.pyplot as plt, random, operator
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self.G = nx.Graph()
        self.G.add_edges_from(read_file(file_path))
        self.label_map = {}
        self.final_groups = []

    def run(self):
        self.initialize_labels()
        self.algorithm(True)

    def run100(self):
        for i in range(100):
            self.initialize_labels()
            self.algorithm(False)
            self.final_groups.append(self.get_unique_groups())
        self.print_results_of_run100()

    def print_results_of_run100(self):
        print("Average number of communities found in 100 iterations: %s" % self.get_average_number_of_groups())
        counted = Counter(self.final_groups)
        for key in counted.keys():
            print("In %d iterations number of communities found was %d" % (counted[key], key))

    def get_unique_groups(self):
        return Counter(self.label_map.values()).__len__()

    def get_average_number_of_groups(self):
        return sum(self.final_groups) / len(self.final_groups)

    def initialize_labels(self):
        for i, node in enumerate(self.G.nodes()):
            self.label_map[node] = i+1

    def draw(self):
        values = [self.label_map.get(node) for node in self.G.nodes()]
        plt.subplot(111)
        nx.draw(self.G, with_labels=self.G.nodes, node_color=values)
        plt.show()

    def find_max_label(self, labels):
        label_count = list(Counter(labels).values())
        if all(label_count[0] == label_cnt for label_cnt in label_count):
            return random.choice(labels)
        else:
            return max(Counter(labels).items(), key=operator.itemgetter(1))[0]

    def algorithm(self, draw):
        if draw:
            self.draw()
        change = True
        while change:
            change = False
            for node in sorted(self.G.nodes()):
                labels = [self.label_map[adj] for adj in self.G[node].keys()]
                new_label = self.find_max_label(labels)
                if self.label_map[node] != new_label:
                    self.label_map[node] = new_label
                    change = True
        if draw:
            self.draw()

def read_file(file_path):
    with open(file_path, "rt") as f:
        data = []
        for line in f:
            line = line.split()
            data.append((line[0], line[1]))
        return data
