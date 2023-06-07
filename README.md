# electricTSP
This is the repository of the paper COMBINING SIMULATION AND NEURAL NETWORKS TO OPTIMIZE THE TSP WITH ELECTRIC VEHICLES IN CONTEXT-AWARE URBAN LOGISTICS.

Authors: Yusef Ahsini, Pablo Díaz-Masa, Belén Inglés, Ana Rubio, Alba Martinez, Aina Magraner and Alberto Conejero.

## Code

- <code> algorithm.py </code> corresponds to an implementation of the algorithm developed in the paper to solve the EV-TSP  for the city of Madrid. It takes as input a list of nodes from the graph of Madrid city <code>madrid_elevation_energy.pckl </code> and the output consists of an ordered list of all the nodes representing the solution to the TSP.
- <code> bellmanFord.py </code> is a Python implementation of the Bellman-Ford algorithm that uses a heap to keep the nodes ordered according to their distances, which can improve efficiency compared to using a list or set to perform edge relaxations. 
- <code> evaluation.py </code> is the scrip that offers the evaluation of the algorithm offered in Tables 1 and 2 in the paper.
- <code> neuralNetworkTraining.py </code> is the scrip used to train and save the Neural Network model using the data generated by <code> simulation.py </code>.
- <code> nn_model_predictor.py </code> is a script where the model trainined in <code> neuralNetworkTraining.py </code> can be used to generate predictions.
- <code> simulation.py </code> is the scrip that simulated the routes trough the months of October and Novemeber 2022 using the data in <code>snapshots_2022.zip</code>. It generates the routes in <code> simulationOctober.csv</code> and <code> simulationNovember.csv</code>
- <code> twoOptNearestNeighnors.py </code> is a Pyhton implementation of the 2-Opt algorithm that uses Nearest Neighbors to generate the initial tour.

## Files

- <code> Madrid{5,10,15}.pkl</code> are the test instances for the city of Madrid. Correspond to python list of list. Each list are a set of stops to visit in the city graph of Madrid (<code>madrid_elevation_energy.pckl</code>)   
- <code>energy_estimation_full.h5</code> is a Keras model trained using <code>nn_model_predictor.py</code> to estimate the energy.
- <code>scaler_full.pkl</code> is the scaler need to use the <code>energy_estimation_full.h5</code> model.
- <code> simu{5,10,15}.pkl</code> are the test instances for the city of Madrid. Correspond to python list of list. Each list are a set of stops to visit in the city graph of Madrid (<code>madrid_elevation_energy.pckl</code>)   
- <code>snapshots_2022.zip</code> are the traffic data for the months of October and November 2022


## Web page
This web page allows the user to specify a starting point and then to add several stops. This returns the optimized route.

URL: https://smartrouteselection.streamlit.app

The main file is the <code> main_web.py </code>, which allows the web page to run.

The files needed to run this .py are: <code>coords.csv</code>, <code>algorithm.py</code> and <code>elev_Madrid.json</code>.

The libraries needed to execute the app are in <code>requirements.txt</code>.

The <code>requirements.txt</code> contains the specific versions of the libraries that allow the streamlit app to run the code without any compatibility problems. (May 2023)

The <code>coords.csv</code> is our own data base of the id and number coordinates of the different nodes, these are specific for the city of Madrid. It has been obtained by the elev_Madrid.json, which was used to modelize the city. In case we want to represent another city, the data format would have to present the same format. The explanation to obtain this json is specified above.

### How it works:
Firstly, we need to obtain the number coordinate of the streets the user has entered. We have used the geocode API in this step.

Next, we obtained the nearest node in our data base, corresponding to the streets the user has entered and we make a list and a dictionary:
 - The list contains the id of de nodes
 - The dictionary has as keys "lat" and "lon" and the coordinates are added in the order the user has entered.

Then, the list is used in the algorithm, since the algorithm is built in order to recieve the id of the nodes and returns a list with the id of the nodes conforming the route and their coordinates.

Finally, with plotly we make the map graphic and we add the traces and specific points. One trace for the outgoing trip and other for the return trip, plus the points refering to the starting point and the stops.




