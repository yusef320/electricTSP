import streamlit as st
import csv
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import datetime
import time
from geopy.geocoders import Nominatim
loc = Nominatim(user_agent="GetLoc")

st.set_page_config(page_title="Título web", layout='wide', initial_sidebar_state = "auto")


def get_time_of_day():
    now = datetime.datetime.now()
    hour = now.hour
    
    if hour >= 6 and hour < 19:
        return "carto-positron"
    else:
        return "carto-darkmatter"


coord_geo = {"Madrid" : {"lon": -3.7025600, "lat": 40.4165000}}

with st.sidebar:
    st.title("SmartRoute")
    lista_paradas = []
    ciudad = "Madrid"
    origen = st.text_input("Origen", "Eje: Gran Vía")
    lista_paradas.append(origen)
    #Opcion de elegir distintas paradas
    n_paradas = st.number_input("Número de paradas a efectuar", step=1, min_value=0)
    
    if n_paradas != 0:
        for i in range(1, n_paradas+1):
            parada = st.text_input(f"Parada {i}:", key = f"{i}")
            lista_paradas.append(parada)
        
    
    ################################

    col1, col2= st.columns(2)
    with col1:
        peso_vehiculo = st.number_input("Inserta el peso del vehículo (Kg)", step=100, min_value=0)
    with col2:
        potencia_motor = st.number_input("Potencia del motor (kWh)", step=10, min_value=0)
    
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
            lista_puntos.append(dirToCoord(lista_coords, getLoc_calle))
    st.write(lista_puntos)

else: pass



#En el data van los puntos de lat y lon de las paradas indicadas 
#(la primera de cada lista es la de origen)
data = {'lat': [], 'lon': []}
df = pd.DataFrame(data)



# Crear el mapa interactivo con Plotly
fig = px.scatter_mapbox(df, lat='lat', lon='lon', center = coord_geo["Madrid"], zoom = 11)

#fig.add_trace(go.Scattermapbox(
#    mode = "markers+lines",
#    lon = data["lon"],
#    lat = data["lat"],
#    marker = {'size': 10}))

fig.update_layout(mapbox_style=theme)
fig.update_layout(height=800,width=850)

# Mostrar el mapa interactivo en Streamlit
st.plotly_chart(fig)


    
    