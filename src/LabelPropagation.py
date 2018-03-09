import networkx as nx, matplotlib.pyplot as plt, random, operator, csv
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self.G = nx.Graph()
        self.G.add_edges_from(read_file(file_path))
        self.val_map = {}
        self.final_groups = []

    def __call__(self, no_of_groups, c, t):
        # write options depending on "c" and "t"
        """"
        self.val_map['A'] = 0
        self.val_map['B'] = 1
        self.val_map['C'] = 0
        self.val_map['D'] = 1
        self.val_map['E'] = 0
        self.val_map['F'] = 1
        self.val_map['G'] = 1
        self.val_map['H'] = 1
        """
        self.initialize_labels(no_of_groups)
        self.run(True)

        for i in range(100):
            self.initialize_labels(no_of_groups)
            self.run(False)
            self.final_groups.append(self.get_unique_groups())
        print("Average number of groups in 100 iterations:", self.get_average_number_of_groups())

    def get_unique_groups(self):
        return Counter(self.val_map.values()).__len__()

    def get_average_number_of_groups(self):
        return sum(self.final_groups) / len(self.final_groups)

    def initialize_labels(self, no_of_groups):
        for i, node in enumerate(self.G.nodes()):
            if no_of_groups > 0:
                self.val_map[node] = random.choice(range(1, no_of_groups + 1))
            else:
                self.val_map[node] = i+1

    def draw(self):
        values = [self.val_map.get(node) for node in self.G.nodes()]
        plt.subplot(111)
        nx.draw(self.G, with_labels=self.G.nodes, node_color=values)
        plt.show()

    def run(self, draw):
        change = True
        while change:
            change = False
            if draw:
                self.draw()
            for node in sorted(self.G.nodes()):
                labels = [self.val_map[adj] for adj in self.G[node].keys()]
                new_label = max(Counter(labels).items(), key=operator.itemgetter(1))[0]
                if self.val_map[node] != new_label:
                    self.val_map[node] = new_label
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


lp = LabelPropagation("../input/inputFile")
lp(4, 0, 0)
