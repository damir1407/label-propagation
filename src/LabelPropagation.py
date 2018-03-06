import networkx as nx, matplotlib.pyplot as plt, random, operator, csv
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path):
        self.G = nx.Graph()
        self.G.add_edges_from(read_file(file_path))
        self.val_map = {}
        self.finalGroups = []

    def __call__(self, noOfGroups, c, t, draw=False):
        # write options depending on "c" and "t"
        for i in range(100):
            self.initializeLabels(noOfGroups)
            self.run(draw)
            self.finalGroups.append(self.getUniqueGroups())
        print(self.getAverageNumberOfGroups())

    def getUniqueGroups(self):
        return Counter(self.val_map.values()).__len__()

    def getAverageNumberOfGroups(self):
        return sum(self.finalGroups) / len(self.finalGroups)

    def initializeLabels(self, noOfGroups):
        for i, node in enumerate(self.G.nodes()):
            if noOfGroups > 0:
                self.val_map[node] = random.choice(range(1, noOfGroups + 1))
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
            for node in self.G.nodes():
                labels = [self.val_map[adj] for adj in self.G[node].keys()]
                newLabel = max(Counter(labels).items(), key=operator.itemgetter(1))[0]
                if self.val_map[node] != newLabel:
                    self.val_map[node] = newLabel
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
