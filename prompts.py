""" i want to model a Graph Convolutional Netwoek (GCN). it has 51 nodes. each node has 3 features.
and each node has a specific name, for example ("N69" , "N70", "N71"). i also have list of edges. list
of edges contains 108  edges. 
list of edges: [('N69', 'N70'), ('N70', 'N71'), ('N71', 'N72'), ('N72', 'N73'), ('N73', 'N74'), ('N73', 'N60'), ('N74', 'N60'), ...]

in this problem, nodes are not labeled. instead, the whole netwrok has a label. the network label categories are 37 labels and the
problem is to predict the network label.
so i expect the output of this GCN to be a vector of size 37 (like softmax). implement the whole network with its loss function,
optimizer and everything.



"""
