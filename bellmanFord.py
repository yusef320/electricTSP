import networkx as nx
import heapq

def bellman_ford_dist(graph, source, target):
    # Initialize distances and predecessors
    dist = {node: float('inf') for node in graph.nodes()}
    dist[source] = 0
    pred = {node: None for node in graph.nodes()}
    heap = [(0, source)]

    # Relax edges repeatedly
    visited = set()
    while heap:
        (d, u) = heapq.heappop(heap)
        if u == target:
            break
        if u in visited:
            continue
        visited.add(u)
        for v, e in graph[u].items():
            cost = dist[u] + e['energy']
            if dist[v] > cost:
                dist[v] = cost
                pred[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist[target]


def bellman_ford_path(graph, source, target, distancia="energy"):
    # Initialize distances and predecessors
    dist = {node: float('inf') for node in graph.nodes()}
    dist[source] = 0
    pred = {node: None for node in graph.nodes()}
    heap = [(0, source)]

    # Relax edges repeatedly
    visited = set()
    while heap:
        (d, u) = heapq.heappop(heap)
        if u == target:
            break
        if u in visited:
            continue
        visited.add(u)
        for v, e in graph[u].items():
            cost = dist[u] + e[distancia]
            if dist[v] > cost:
                dist[v] = cost
                pred[v] = u
                heapq.heappush(heap, (dist[v], v))

    # Build the path from source to target
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred[node]
    path.reverse()

    return path, dist[target]