import networkx as nx
import heapq

def bellman_ford_dist(graph, source, target):
    """
    Returns the minimum energy consumption between two nodes
    in the city graph
    """
    return nx.bellman_ford_path_length(G, graph, target, weight='energy')


def bellman_ford_path(graph, source, target, weight="energy"):
    """
    Returns the minimum energy consumption between two nodes
    in the city graph and the path
    """
    path = nx.bellman_ford_path(graph, source, target)
    energy = 0
    for i in range(len(path) - 1):
        energy += graph[path[i]][path[i+1]][weight]
    return path, energy
