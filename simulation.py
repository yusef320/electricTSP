import math
import pandas as pd
import os
import json
import networkx as nx
import random
import time
import bellmanFord
import numpy as np


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



#Load Madrid city graph with altitude
G = nx.read_gpickle("Madrid_elevation.gpickle")
nodes = list(G.nodes)
data = G.nodes(data=True)


#GRAPH PREPROCESING

#some roads have more than one max speed, we compute the average
for i, v, attr in G.edges(data=True):
    try:
        if type(attr["maxspeed"]) == list:
            attr["maxspeed"] = sum(int(n) for n in attr["maxspeed"]) / len(attr["maxspeed"])
    except:
        pass

#for the roads without a speed, we impute it with the average of their road type
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

#in case in the graph exists a node without altitude, we impute it with nearest neighbors
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

#obtain road gradient in radiants
for u, v, attr in G.edges(data=True):
    attr["gradient"] = float(math.atan((data[v]["elevation"] - data[u]["elevation"]) / attr["length"]))
#estimate energy needed for each edge
for u, v, attr in G.edges(data=True):
    attr["energy"] = estimate_energy(attr["maxspeed"], attr["gradient"], attr["length"])

#GENERATE THE DATA FOR NOVEMBER 2022
simulation_data = []
for shot in os.listdir(f'./snapshots_2022_11'):  # quitar [:2] para iterar las 720 snapshots
    grafo = G.copy()
    with open(f'./snapshots_2022_11/{shot}', 'r') as fp:
        snapshot = json.load(fp)
    for a, u, attr in grafo.edges(data=True):
        if str(attr["osmid"]) in snapshot:
            if np.isnan(float(snapshot[str(attr["osmid"])])):
                pass
            else:
                attr["maxspeed"] = float(snapshot[str(attr["osmid"])])
                attr["energy"] = estimate_energy(attr["maxspeed"], attr["gradient"], attr["length"])

    date = shot.replace(".json", "").split("_")[3:]
    start = time.time()
    for i in range(200):
        stops = random.sample(nodes, 2)
        l = date.copy()
        l = l + [data[stops[0]]["x"], data[stops[0]]["y"], data[stops[0]]["elevation"]]
        l = l + [data[stops[1]]["x"], data[stops[1]]["y"], data[stops[1]]["elevation"]]
        energy = bellmanFord.bellman_ford_dist(grafo, source=stops[0], target=stops[1])
        l.append(energy)
        simulation_data.append(l)

    print("Time: ", time.time() - start, "s")

df = pd.DataFrame(simulation_data,
                  columns=["Day", "Hour", "org_lon", "org_lat", "org_alt", "dest_lon", "dest_lat", "dest_alt",
                           "energy_used"])

df.to_csv("simulationNovember.csv", sep=";", index=False)

#GENERATE THE DATA FOR OCTOBER 2022
simulation_data = []
for shot in os.listdir(f'./snapshots_2022_10'):  # quitar [:2] para iterar las 720 snapshots
    grafo = G.copy()
    with open(f'./snapshots_2022_10/{shot}', 'r') as fp:
        snapshot = json.load(fp)
    for a, u, attr in grafo.edges(data=True):
        if str(attr["osmid"]) in snapshot:
            if np.isnan(float(snapshot[str(attr["osmid"])])):
                pass
            else:
                attr["maxspeed"] = float(snapshot[str(attr["osmid"])])
                attr["energy"] = estimate_energy(attr["maxspeed"], attr["gradient"], attr["length"])

    date = shot.replace(".json", "").split("_")[3:]
    start = time.time()
    for i in range(200):
        stops = random.sample(nodes, 2)
        l = date.copy()
        l = l + [data[stops[0]]["x"], data[stops[0]]["y"], data[stops[0]]["elevation"]]
        l = l + [data[stops[1]]["x"], data[stops[1]]["y"], data[stops[1]]["elevation"]]
        energy = bellmanFord.bellman_ford_dist(grafo, source=stops[0], target=stops[1])
        l.append(energy)
        simulation_data.append(l)

    print("Time: ", time.time() - start, "s")


df = pd.DataFrame(simulation_data,
                  columns=["Day", "Hour", "org_lon", "org_lat", "org_alt", "dest_lon", "dest_lat", "dest_alt",
                           "energy_used"])
df.to_csv("simulationOctober.csv", sep=";", index=False)

print("Simulation finished, data saved.")