
import networkx as nx
import numpy as np
from nn_model_predictor import neural_network_energy_estimation
from bellmanFord import bellman_ford_path
from twoOptNearestNeigbors import RouteFinder
from datetime import datetime

def electric_tsp(stops):
    f = open('madrid_elevation_energy.pckl', 'rb')
    G = pickle.load(f)
    f.close()
    grafo_tsp = nx.Graph()
    date_today = datetime.now()
    grafo_tsp.add_nodes_from(stops)
    weekday = date_today.strftime("%A")
    working = 1
    if weekday in ["Saturday", "Sunday"]:
        working = 0
    for i in stops:
        for j in stops:
            if i != j:
                d_temp = neural_network_energy_estimation(i, j, working, str(date_today.hour), weekday, G)
                grafo_tsp.add_edge(i, j, weight=d_temp)
    mat = nx.adjacency_matrix(grafo_tsp).todense()
    mat = np.array(mat).reshape(-1, mat.shape[1])
    for i in range(len(stops)):
        mat[i][i] = np.inf
    route_finder = RouteFinder(mat, stops, iterations=3000)
    best_distance, best_route = route_finder.solve()
    ruta_final = []
    for i in range(len(best_route) - 1):
        r_temp, dist = bellman_ford_path(G, best_route[i], best_route[i + 1])
        ruta_final = ruta_final + r_temp[1:]

    return ruta_final






