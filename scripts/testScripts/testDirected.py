import networkx as nx
import matplotlib.pyplot as plt


undirectedGraph = nx.erdos_renyi_graph(n=4, p=0.5)
directedGraph = nx.erdos_renyi_graph(n=4, p=0.5, directed=True)

nx.draw(undirectedGraph)
plt.savefig("ugraph.png")