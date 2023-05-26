# electricTSP
This is the repository of the paper ELECTRIC VEHICLE TSP OPTIMIZATION IN URBAN LOGISTICS: A TOPOGRAPHY AND TRAFFIC-AWARE APPROACH.

Authors: Yusef Ahsini, Pablo Díaz-Masa, Belén Inglés, Ana Rubio, Alba Martinez, Aina Magraner and Alberto Conejero.

# Web page
This web pages allows the user to specify a starting point and then to add several stops. This returns the optimized route.
URL: https://smartrouteselection.streamlit.app

The main file is the main_web.py, which allows the web page to run.
The files needed to run this .py are: requirements.txt, coords.csv, algorithm.py and elev_Madrid.json.
The libraries are: streamlit, csv, plotly, pandas and geopy.geocoders.

The requirements.txt contains the specific versions of the libraries that allow the streamlit app to run the code without any compatibility problems.

The coords.csv is our own data base of the id and number coordinates of the different nodes, these are specific for the city of Madrid. It has been obtained by the elev_Madrid.json, which was used to modelize the city. In case we want to represent another city, the data format would have to present the same format. The explanation to obtain this json is specified above.

The algorithm.py is the algorithm created in this project, used to calculate the optimized route.

## How it works:
Firstly, we need to obtain the number coordinate of the streets the user has entered. We have used the geocode API in this step.
Next, we obtained the nearest node in our data base, corresponding to the streets the user has entered and we make a list and a dictionary:
 - The list contains the id of de nodes
 - The dictionary has as keys "lat" and "lon" and the coordinates are added in the order the user has entered.

Then, the list is used in the algorithm, since the algorithm is built in order to recieve the id of the nodes and returns a list with the id of the nodes conforming the route and their coordinates.
Finally, with plotly we make the map graphic and we add the traces and specific points. One trace for the outgoing trip and other for the return trip, plus the points refering to the starting point and the stops.




