import networkx as nx
import matplotlib.pyplot as plt
import random
import operator
from collections import Counter


class LabelPropagation:
    def __init__(self):
        self.G = nx.Graph()
        self.G.add_edges_from(
            [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D'),
             ('B', 'E'), ('B', 'F'), ('E', 'F'), ('F', 'G'), ('E', 'I'), ('I', 'J'),
             ('G', 'J'), ('E', 'G'), ('E', 'J'), ('F', 'I'), ('F', 'J'), ('I', 'G'),
             ('C', 'H'), ('H', 'K'), ('H', 'L'), ('K', 'L'), ('K', 'J')])

        groups = [1.0, 0.75, 0.5, 0.0]
        self.val_map = {}
        for node in self.G.nodes():
            self.val_map[node] = random.choice(groups)

    def draw(self):
        values = [self.val_map.get(node) for node in self.G.nodes()]
        plt.subplot(111)
        nx.draw(self.G, with_labels=self.G.nodes, node_color=values)
        plt.show()

    def run(self):
        change = True
        while change:
            change = False
            self.draw()
            for node in self.G.nodes():
                print(node)
                labels = [self.val_map[adj] for adj in self.G[node].keys()]
                newLabel = max(Counter(labels).items(), key=operator.itemgetter(1))[0]
                if self.val_map[node] != newLabel:
                    self.val_map[node] = newLabel
                    change = True

lp = LabelPropagation()
lp.run()
lp.draw()
