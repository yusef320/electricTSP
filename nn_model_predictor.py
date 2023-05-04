import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model, save_model
import pickle
from sklearn.preprocessing import MinMaxScaler
model= load_model('energy_estimation_full.h5', compile=False)
with open('scaler_full.pkl', 'rb') as f:
    scaler = pickle.load(f)

def neural_network_energy_estimation(a, b, workingday,hour, weekday, G, model, scaler):
    input_var = ['org_lon', 'org_lat', 'org_alt', 'dest_lon', 'dest_lat', 'dest_alt',"Working_day",
           'Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
           'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
           'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18',
           'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 'Weekday_Friday',
           'Weekday_Monday', 'Weekday_Saturday', 'Weekday_Sunday',
           'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday']

    model_input = [0 for _ in input_var]
    model_input[0] = G.nodes(data=True)[a]["x"]
    model_input[1] = G.nodes(data=True)[a]["y"]
    model_input[2] = G.nodes(data=True)[a]["elevation"]
    model_input[3] = G.nodes(data=True)[b]["x"]
    model_input[4] = G.nodes(data=True)[b]["y"]
    model_input[5] = G.nodes(data=True)[b]["elevation"]
    model_input[6] = workingday
    model_input[input_var.index("Hour_"+hour)] = 1
    model_input[input_var.index("Weekday_"+weekday)] = 1
    model_input = scaler.transform(pd.DataFrame([model_input], columns=input_var))
    return float(model.__call__(tf.constant(model_input)))

