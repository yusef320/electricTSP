
import streamlit as st
import csv
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import datetime
import time
from algorithm import electric_tsp
from geopy.geocoders import Nominatim
from time import sleep
loc = Nominatim(user_agent="GetLoc")

st.set_page_config(page_title="SmartRoute", layout='wide', initial_sidebar_state = "auto")
data = {'lat': [], 'lon': []}

def get_time_of_day():
    now = datetime.datetime.now()
    hour = now.hour
    
    if hour >= 6 and hour < 19:
        return "carto-positron"
    else:
        return "carto-darkmatter"


coord_geo = {"Madrid" : {"lon": -3.7025600, "lat": 40.4165000}}

with st.sidebar:
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.title("SmartRoute")
    lista_paradas = []
    ciudad = "Madrid"
    origen = st.text_input("Origen", "Ejemplo: Gran Vía")
    lista_paradas.append(origen)
    #Opcion de elegir distintas paradas
    n_paradas = st.number_input("Número de paradas a efectuar", step=1, min_value=0)
    if n_paradas != 0:
        for i in range(1, n_paradas+1):
            parada = st.text_input(f"Parada {i}:", key = f"{i+10}")
            lista_paradas.append(parada)
                 
                    
    ################################

    # Insertar un espacio en blanco para poner "elegir tema" abajo del todo
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
  

    col3, col4= st.columns(2)
    with col3:
        if get_time_of_day() == "carto-positron": ind = 0
        else: ind = 1
        theme2 = st.selectbox("Elige el tema", ("Light", "Dark"), index = ind)
        if theme2 == "Light":
            theme = "carto-positron"
        else:
            theme = "carto-darkmatter"
    

    ################################################################
df_puntos = {'lat':[], 'lon':[]}
lista_coords = pd.read_csv("coords.csv", header = None)
lista_coords = lista_coords.iloc[:,0:3]
if origen != "Eje: Gran Vía":
    def dirToCoord(df, calle):
        minimo = 10000
        for i in range(len(df)):
            eje_lon = abs(df.loc[i,1]-calle.longitude)
            eje_lat = abs(df.loc[i,2]-calle.latitude)
            a = eje_lat + eje_lon
            if a < minimo:
                minimo = a
                fila = i
        return lista_coords.iloc[fila,0]

    lista_puntos = []
    
    for u in lista_paradas:
        if "Madrid, España" not in u:
            u = u + ", Madrid, España"
            getLoc_calle = loc.geocode(u)
            if getLoc_calle:
                lista_puntos.append(dirToCoord(lista_coords, getLoc_calle))
                a = lista_coords.loc[lista_coords.loc[:,0] == dirToCoord(lista_coords, getLoc_calle)]
                df_puntos['lat'].append(float(a[2].values))
                df_puntos['lon'].append(float(a[1].values))
                                                                         
            else:
                pass
    if len(lista_puntos) > 1:
        ruta = electric_tsp(lista_puntos)

        df_1 = {'lon': [],'lat':[]}
        df_vuelta = {'lon': [],'lat':[]}
        fin = ruta.index(lista_puntos[-1])
        ruta_1 = ruta[:fin]
        ruta_2 = ruta[fin:]
        for p in range(len(ruta_1)):
            a = lista_coords.loc[lista_coords.loc[:,0] == ruta_1[p]]
            df_1['lon'].append(float(a[1].values))
            df_1['lat'].append(float(a[2].values))     
            
        for p in range(len(ruta_2)):
            a = lista_coords.loc[lista_coords.loc[:,0] == ruta_2[p]]
            df_vuelta['lon'].append(float(a[1].values))
            df_vuelta['lat'].append(float(a[2].values))    

        fig = go.Figure(go.Scattermapbox(
            mode='markers+text',
            name = 'ORIGEN',
            lat=[df_puntos['lat'][0]],
            lon=[df_puntos['lon'][0]],
            marker = {'size': 9, 'color':'red'},
            ))
        if len(lista_puntos) == 2:
            sleep(1)
        else:
            pass
        fig.add_trace(go.Scattermapbox(
            mode = 'markers+text',
            name = 'PARADAS',
                lat=df_puntos['lat'][1:],
                lon=df_puntos['lon'][1:],
                marker = {'size': 9, 'color':'black'},
                textposition='top right',
                textfont=dict(size=9, color='black'),
                text = lista_paradas,
                hoverinfo='text'))

        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            name = 'IDA',
            lon = df_1["lon"],
            lat = df_1["lat"],
            marker=dict(color='green', size=4)))

        fig.add_trace(go.Scattermapbox(
            mode = "lines",
            name = 'VUELTA',
            lon = df_vuelta["lon"],
            lat = df_vuelta["lat"],
            marker=dict(color='royalblue', size=4)))
    else:
        fig = px.scatter_mapbox(df_puntos, lat='lat', lon='lon', center = coord_geo["Madrid"], zoom = 11)
        
    fig.update_layout(mapbox_style=theme, mapbox=dict(center = coord_geo["Madrid"], zoom = 11))
    fig.update_layout(height=900,width=1000)  
    
else: 
    fig = px.scatter_mapbox(df_puntos, lat='lat', lon='lon', center = coord_geo["Madrid"], zoom = 11)
    fig.update_layout(mapbox_style=theme)
    fig.update_layout(height=900,width=1000)



# Mostrar el mapa interactivo en Streamlit
st.plotly_chart(fig)


    
    
