from labelpropagation.label_propagation import LabelPropagation

lp = LabelPropagation("input-examples/ucidata-zachary/out.ucidata-zachary", "U")
lp.draw_graph()

number_of_communities = lp.run(label_resolution="retention", equilibrium="change", order="asynchronous", weighted=False)
print(number_of_communities)
lp.draw_graph()

number_of_communities = lp.consensus_clustering(label_resolution="retention", equilibrium="change", order="asynchronous"
                                                , threshold=6, number_of_partitions=12, weighted=False)
print(number_of_communities)
lp.draw_graph()
