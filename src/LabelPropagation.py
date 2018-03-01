import networkx as nx, matplotlib.pyplot as plt, random, operator, csv
from collections import Counter


class LabelPropagation:
    def __init__(self, file_path, method, noOfGroups):
        self.G = nx.Graph()
        self.G.add_edges_from(read_file(file_path))

        self.val_map = {}
        count=1
        for node in self.G.nodes():
            if method == "random":
                self.val_map[node] = random.choice(range(count, noOfGroups+count))
            elif method == "unique":
                self.val_map[node] = count
                count = count + 1

    def __call__(self, c, t):
        # write options depending on "c" and "t"
        self.run()

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
                labels = [self.val_map[adj] for adj in self.G[node].keys()]
                newLabel = max(Counter(labels).items(), key=operator.itemgetter(1))[0]
                if self.val_map[node] != newLabel:
                    self.val_map[node] = newLabel
                    change = True

def read_file(file_path):
    f = open(file_path, "rt")
    data = []
    for line in csv.reader(f, delimiter=" "):
        data.append((line[0], line[1]))
    return data

lp = LabelPropagation("../input/inputFile", "random", 4)
lp(1,1)
lp.draw()
