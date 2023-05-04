import networkx as nx
import json
import os
import random
import time
import datetime
import numpy as np
from nn_model_predictor import neural_network_energy_estimation
from bellmanFord import bellman_ford_dist,bellman_ford_path
from twoOptNearestNeigbors import RouteFinder
import math
import warnings
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

"""
In this .py is available the code to obtain the results in Table 1 and 2 and Figure 7
"""

def estimate_energy(v, gradient, length):
    """
    :param v: maxspeed in km/h of the road
    :param gradient: gradient in radians of the road
    :param length: length in meters of the road
    :return:
    """
    speed= float(v)/3.6 #from km/h to m/s
    ft=1480*9.8*math.sin(gradient)+1480*9.8*math.cos(gradient)*0.01+((1.24*3.26*0.31)/2)*speed**2
    if ft>=0:
        P = ((ft*speed)/0.95) +500
    else:
        P = (ft*speed*0.6) +500
    e_dt=3*10**(-10)*v**6 - 2*10**(-7)*v**5 +5*10**(-5)*v**4 -0.0057*v**3  +0.358*v**2-10.26*v+ 139.27
    e_aux=121.1*v**(-0.794)
    return(P*((length/1000)/v)+ e_dt*(length/1000)+ e_aux*(length/1000))


G = nx.read_gpickle("Madrid_elevation.gpickle")
nodes = list(G.nodes)
data = G.nodes(data=True)

for i, v, attr in G.edges(data=True):
    try:
        if type(attr["maxspeed"]) == list:
            attr["maxspeed"] = sum(int(n) for n in attr["maxspeed"]) / len(attr["maxspeed"])
    except:
        pass

veloc_vias = {
    'residential': 40.61,
    'tertiary': 46.37,
    'motorway_link': 60.13,
    'primary': 48.90,
    'unclassified': 41.33,
    'motorway': 76.24,
    'secondary': 47.42,
    'living_street': 23.20,
    'trunk_link': 54.00,
    'tertiary_link': 45.75,
    'primary_link': 49.32,
    'secondary_link': 48.88,
    'trunk': 53.95,
    'residential_link': 40.61}

for u, v, attr in G.edges(data=True):
    try:
        attr["maxspeed"] = float(attr["maxspeed"])
    except:
        if type(attr["highway"]) == list:
            attr["maxspeed"] = veloc_vias[attr["highway"][0]]
        else:
            attr["maxspeed"] = veloc_vias[attr["highway"]]

no_altitud = ["None"]
while len(no_altitud) > 0:
    no_altitud = []
    for i in range(len(nodes)):
        try:
            data[nodes[i]]["elevation"]
        except:
            for edge in G.successors(nodes[i]):
                try:
                    data[nodes[i]]["elevation"] = data[edge]["elevation"]
                except:
                    no_altitud.append("None")

for u, v, attr in G.edges(data=True):
    attr["gradient"] = float(math.atan((data[v]["elevation"] - data[u]["elevation"]) / attr["length"]))

for u, v, attr in G.edges(data=True):
    attr["energy"] = estimate_energy(attr["maxspeed"], attr["gradient"], attr["length"])

nn_distance = []
nn_time = []
bf_distance = []
bf_time = []

for instance in range(50):
    spname = random.choice(os.listdir(f'./snapshots_2022_10') + os.listdir(f'./snapshots_2022_11'))
    stops = random.sample(nodes, 10)
    grafo = G.copy()
    try:
        with open(f'./snapshots_2022_10/{spname}', 'r') as fp:
            snapshot = json.load(fp)
    except:
        with open(f'./snapshots_2022_11/{spname}', 'r') as fp:
            snapshot = json.load(fp)

    for a, u, attr in grafo.edges(data=True):
        if str(attr["osmid"]) in snapshot:
            if np.isnan(float(snapshot[str(attr["osmid"])])):
                pass
            else:
                attr["maxspeed"] = float(snapshot[str(attr["osmid"])])
                attr["energy"] = estimate_energy(attr["maxspeed"], attr["gradient"], attr["length"])

    date = spname.replace(".json", "").split("_")
    start = time.time()
    grafo_tsp = nx.Graph()
    grafo_tsp.add_nodes_from(stops)
    weekday = datetime.date(2023, int(date[2]), int(date[3])).strftime("%A")
    working = 1
    if weekday in ["Saturday", "Sunday"] or f"{date[2]}/{date[3]}" in ["10/12", "11/1", "11/"]:
        working = 0
    for i in stops:
        for j in stops:
            if i != j:
                d_temp = neural_network_energy_estimation(i, j, working, date[4], weekday, grafo)
                grafo_tsp.add_edge(i, j, weight=d_temp)
    mat = nx.adjacency_matrix(grafo_tsp).todense()
    mat = np.array(mat).reshape(-1, mat.shape[1])
    for i in range(len(stops)):
        mat[i][i] = np.inf
    route_finder = RouteFinder(mat, stops, iterations=1000)
    best_distance, best_route = route_finder.solve()
    ruta_final = []
    dist_final = 0
    for i in range(len(best_route) - 1):
        r_temp, dist = bellman_ford_path(grafo, best_route[i], best_route[i + 1])
        dist_final += dist
        ruta_final = ruta_final + r_temp
    nn_time.append(time.time() - start)

    nn_distance.append(dist_final)
    start = time.time()
    grafo_tsp = nx.Graph()
    grafo_tsp.add_nodes_from(stops)
    for i in stops:
        for j in stops:
            if i != j:
                d_temp = bellman_ford_dist(grafo, i, j)
                grafo_tsp.add_edge(i, j, weight=d_temp)
    mat = nx.adjacency_matrix(grafo_tsp).todense()
    mat = np.array(mat).reshape(-1, mat.shape[1])
    for i in range(len(stops)):
        mat[i][i] = np.inf
    route_finder = RouteFinder(mat, stops, iterations=1000)
    best_distance, best_route = route_finder.solve()
    ruta_final = []
    dist_final = 0
    for i in range(len(best_route) - 1):
        r_temp, dist = bellman_ford_path(grafo, best_route[i], best_route[i + 1])
        dist_final += dist
        ruta_final = ruta_final + r_temp
    bf_time.append(time.time() - start)
    bf_distance.append(dist_final)
print("Table 1")
print(f"NN: average energy {sum(nn_distance)/len(nn_distance)}W\taverage time {sum(nn_time)/len(nn_time)}")
print(f"Bellman Ford: average energy {sum(bf_distance)/len(bf_distance)}W\taverage time {sum(bf_time)/len(bf_time)}")

dist_final = {5: [[], []], 10: [[], []], 15: [[], []]}
time_final = {5: [[], []], 10: [[], []], 15: [[], []]}
energy_final = {5: [[], []], 10: [[], []], 15: [[], []]}

with open('madrid5.pkl', 'rb') as f:
    madrid5 = pickle.load(f)
with open('madrid10.pkl', 'rb') as f:
    madrid10 = pickle.load(f)
with open('madrid15.pkl', 'rb') as f:
    madrid15 = pickle.load(f)

spname = 'snapshot_2022_10_10_14.json'
grafo = G.copy()
with open(f'./snapshots_2022_10/{spname}', 'r') as fp:
    snapshot = json.load(fp)

for a, u, attr in grafo.edges(data=True):
    if str(attr["osmid"]) in snapshot:
        if np.isnan(float(snapshot[str(attr["osmid"])])):
            pass
        else:
            attr["maxspeed"] = float(snapshot[str(attr["osmid"])])
            attr["energy"] = estimate_energy(attr["maxspeed"], attr["gradient"], attr["length"])

for index, istance in enumerate([madrid5, madrid10, madrid15]):
    print(f"Instance Madrid{(index + 1) * 5}")
    for n_subinstance, stops in enumerate(istance):
        print(f"Route {n_subinstance}")
        date = spname.replace(".json", "").split("_")
        grafo_tsp = nx.Graph()
        grafo_tsp.add_nodes_from(stops)
        weekday = datetime.date(2022, int(date[2]), int(date[3])).strftime("%A")
        working = 1
        if weekday in ["Saturday", "Sunday"] or f"{date[2]}/{date[3]}" in ["10/12", "11/1", "11/"]:
            working = 0
        for i in stops:
            for j in stops:
                if i != j:
                    d_temp = neural_network_energy_estimation(i, j, working, date[4], weekday, grafo)
                    grafo_tsp.add_edge(i, j, weight=d_temp)
        mat = nx.adjacency_matrix(grafo_tsp).todense()
        mat = np.array(mat).reshape(-1, mat.shape[1])
        for i in range(len(stops)):
            mat[i][i] = np.inf
        route_finder = RouteFinder(mat, stops, iterations=1000)
        best_distance, best_route = route_finder.solve()
        ruta_final = []
        for i in range(len(best_route) - 1):
            r_temp, dist = bellman_ford_path(grafo, best_route[i], best_route[i + 1])
            ruta_final = ruta_final + r_temp
        distance = 0
        time = 0
        energy = 0
        for i in range(len(ruta_final) - 1):
            if ruta_final[i] != ruta_final[i + 1]:
                edge_data = grafo.get_edge_data(ruta_final[i], ruta_final[i + 1])
                distance += edge_data["length"]
                time += ((edge_data["length"] / 1000) / edge_data["maxspeed"]) * 60  # minutes
                energy += edge_data["energy"]

        print(f"Our algorithm, Distance: {round(distance)} m, time: {round(time)} min, energy: {round(energy)} W")
        dist_final[(index + 1) * 5][0].append(distance)
        time_final[(index + 1) * 5][0].append(time)
        energy_final[(index + 1) * 5][0].append(energy)

        grafo_tsp = nx.Graph()
        grafo_tsp.add_nodes_from(stops)
        for i in stops:
            for j in stops:
                if i != j:
                    d_temp = nx.dijkstra_path_length(grafo, i, j, weight='length')
                    grafo_tsp.add_edge(i, j, weight=d_temp)

        mat = nx.adjacency_matrix(grafo_tsp).todense()
        mat = np.array(mat).reshape(-1, mat.shape[1])
        for i in range(len(stops)):
            mat[i][i] = np.inf
        route_finder = RouteFinder(mat, stops, iterations=1000)
        best_distance, best_route = route_finder.solve()
        ruta_final = []
        for i in range(len(best_route) - 1):
            r_temp = nx.shortest_path(grafo, best_route[i], best_route[i + 1], weight='length')
            ruta_final = ruta_final + r_temp

        distance = 0
        time = 0
        energy = 0
        for i in range(len(ruta_final) - 1):
            if ruta_final[i] != ruta_final[i + 1]:
                edge_data = grafo.get_edge_data(ruta_final[i], ruta_final[i + 1])
                distance += edge_data["length"]
                time += ((edge_data["length"] / 1000) / edge_data["maxspeed"]) * 60  # minutes
                energy += edge_data["energy"]
        print(f"Minimizing distance, Distance: {round(distance)} m, time: {round(time)} min, energy: {round(energy)} W")
        dist_final[(index + 1) * 5][1].append(distance)
        time_final[(index + 1) * 5][1].append(time)
        energy_final[(index + 1) * 5][1].append(energy)
        print("")


dist_final= {5:[[],[]], 10:[[],[]], 15:[[],[]]}
time_final= {5:[[],[]], 10:[[],[]], 15:[[],[]]}
energy_final= {5:[[],[]], 10:[[],[]], 15:[[],[]]}

with open('madrid5.pkl', 'rb') as f:
    madrid5 = pickle.load(f)
with open('madrid10.pkl', 'rb') as f:
    madrid10 = pickle.load(f)
with open('madrid15.pkl', 'rb') as f:
    madrid15 = pickle.load(f)


spname= 'snapshot_2022_10_10_14.json'
grafo = G.copy()
with open(f'./snapshots_2022_10/{spname}', 'r') as fp:
    snapshot = json.load(fp)

for a,u, attr in grafo.edges(data=True):
    if str(attr["osmid"]) in snapshot:
        if np.isnan(float(snapshot[str(attr["osmid"])])):
            pass
        else:
            attr["maxspeed"] = float(snapshot[str(attr["osmid"])])
            attr["energy"] = estimate_energy(attr["maxspeed"],attr["gradient"],attr["length"])

for index,istance in enumerate([madrid5,madrid10,madrid15]):
    for n_subinstance,stops in enumerate(istance):
        date = spname.replace(".json","").split("_")
        grafo_tsp = nx.Graph()
        grafo_tsp.add_nodes_from(stops)
        weekday = datetime.date(2022,int(date[2]),int(date[3])).strftime("%A")
        working= 1
        if weekday in ["Saturday", "Sunday"] or f"{date[2]}/{date[3]}" in ["10/12","11/1","11/"]:
            working=0
        for i in stops:
            for j in stops:
                if i!=j:
                    d_temp = neural_network_energy_estimation(i, j, working,date[4],weekday, grafo)
                    grafo_tsp.add_edge(i, j, weight=d_temp)
        mat = nx.adjacency_matrix(grafo_tsp).todense()
        mat = np.array(mat).reshape(-1, mat.shape[1])
        for i in range(len(stops)):
            mat[i][i] = np.inf
        route_finder = RouteFinder(mat, stops, iterations=1000)
        best_distance, best_route = route_finder.solve()
        ruta_final=[]
        for i in range(len(best_route)-1):
            r_temp, dist = bellman_ford_path(grafo, best_route[i], best_route[i+1])
            ruta_final = ruta_final+r_temp
        distance= 0
        time= 0
        energy= 0
        for i in range(len(ruta_final)-1):
            if ruta_final[i] != ruta_final[i+1]:
                edge_data = grafo.get_edge_data(ruta_final[i], ruta_final[i+1])
                distance += edge_data["length"]
                time += ((edge_data["length"]/1000)/edge_data["maxspeed"])*60 #minutes
                energy += edge_data["energy"]

        dist_final[(index+1)*5][0].append(distance)
        time_final[(index+1)*5][0].append(time)
        energy_final[(index+1)*5][0].append(energy)

        grafo_tsp = nx.Graph()
        grafo_tsp.add_nodes_from(stops)
        for i in stops:
            for j in stops:
                if i!=j:
                    d_temp = nx.dijkstra_path_length(grafo, i, j, weight='length')
                    grafo_tsp.add_edge(i, j, weight=d_temp)

        mat = nx.adjacency_matrix(grafo_tsp).todense()
        mat = np.array(mat).reshape(-1, mat.shape[1])
        for i in range(len(stops)):
            mat[i][i] = np.inf
        route_finder = RouteFinder(mat, stops, iterations=1000)
        best_distance, best_route = route_finder.solve()
        ruta_final=[]
        for i in range(len(best_route)-1):
            r_temp = nx.shortest_path(grafo, best_route[i], best_route[i+1], weight='length')
            ruta_final = ruta_final+r_temp

        distance= 0
        time= 0
        energy= 0
        for i in range(len(ruta_final)-1):
            if ruta_final[i] != ruta_final[i+1]:
                edge_data = grafo.get_edge_data(ruta_final[i], ruta_final[i+1])
                distance += edge_data["length"]
                time += ((edge_data["length"]/1000)/edge_data["maxspeed"])*60 #minutes
                energy += edge_data["energy"]
        dist_final[(index+1)*5][1].append(distance)
        time_final[(index+1)*5][1].append(time)
        energy_final[(index+1)*5][1].append(energy)

print("Table 2")
print("Average distance (m)")
print("Instance\tEnergy\tDistance")
print("Madird5",sum(dist_final[5][0])/25, sum(dist_final[5][1])/25)
print("Madrid10",sum(dist_final[10][0])/25, sum(dist_final[10][1])/25)
print("Madrid15",sum(dist_final[15][0])/25, sum(dist_final[15][1])/25)
print("")
print("Average time to finish the route (min)")
print("Instance\tEnergy\tDistance")
print("Madird5",sum(time_final[5][0])/25, sum(time_final[5][1])/25)
print("Madird10",sum(time_final[10][0])/25, sum(time_final[10][1])/25)
print("Madird15",sum(time_final[15][0])/25, sum(time_final[15][1])/25)
print("")
print("Average energy (Wh)")
print("Instance\tEnergy\tDistance")
print("Madird5",sum(energy_final[5][0])/25, sum(energy_final[5][1])/25)
print("Madird10",sum(energy_final[10][0])/25, sum(energy_final[10][1])/25)
print("Madird15",sum(energy_final[15][0])/25, sum(energy_final[15][1])/25)

#Comparison between optimizing ditance and energy, Figure 7

org = 857175705  # Terminal 2 Airport
dest = 29434207  # Campus Sur UPM
ruta_dist = nx.shortest_path(G, org, dest, weight='length')
ruta_energ, dist = bellman_ford_path(G, org, dest)
data_dist = [[0], [data[org]["elevation"]], [0]]  # dist, altura, energy
data_energ = [[0], [data[org]["elevation"]], [0]]  # dist, altura, energy
for i in range(len(ruta_dist) - 1):
    data_dist[1].append(data[ruta_dist[i + 1]]["elevation"])
    edge_data = G.get_edge_data(ruta_dist[i], ruta_dist[i + 1])
    data_dist[2].append(edge_data["energy"] + data_dist[2][i])
    data_dist[0].append(edge_data["length"] + data_dist[0][i])

for i in range(len(ruta_energ) - 1):
    data_energ[1].append(data[ruta_energ[i + 1]]["elevation"])
    edge_data = G.get_edge_data(ruta_energ[i], ruta_energ[i + 1])
    data_energ[2].append(edge_data["energy"] + data_energ[2][i])
    data_energ[0].append(edge_data["length"] + data_energ[0][i])

fig, ax1 = plt.subplots()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
ax1.plot(data_energ[0], data_energ[1], '#12F460', label='Eficient route node elevation')
ax1.plot(data_dist[0], data_dist[1], '#A80600', label='Short route node elevation')
ax1.set_xlabel('Distance traveled')
ax1.set_ylabel('Elevation')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # segundo eje y

ax2.plot(data_energ[0], data_energ[2], '#06A83F', label='Eficient route cum. cons.')
ax2.plot(data_dist[0], data_dist[2], '#F45D0A', label='Short route cum. cons.')
ax2.set_ylabel('Cumulative consumption')
ax2.tick_params(axis='y')
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.39, 0.4, 0.5))
fig.set_size_inches(10, 5)
plt.rcParams.update({'font.size': 11, 'font.family': 'Arial'})
# plt.savefig('mi_grafico.png', dpi=300)
plt.show()