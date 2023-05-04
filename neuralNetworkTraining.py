import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model, save_model
import pickle

#load the data
df_october = pd.read_csv("simulationOctober.csv",sep=";")
df_november = pd.read_csv("simulationNovember.csv",sep=";")

#create a column with the weekday
df_october["Weekday"] = df_october["Day"].apply(lambda x: datetime.date(2022,10,int(x)).strftime("%A"))
df_november["Weekday"] = df_november["Day"].apply(lambda x: datetime.date(2022,10,int(x)).strftime("%A"))

#create a binary column if it is a working day. November 1st and October 12th are holidays in Spain
df_october["Working_day"] = df_october.apply(lambda x: x.Day !="12" and x.Weekday not in ["Saturday", "Sunday"], axis=1)
df_november["Working_day"] = df_november.apply(lambda x: x.Day !="1" and x.Day !="9" and x.Weekday not in ["Saturday", "Sunday"], axis=1)

#join both months
df = pd.concat([df_october, df_november], ignore_index =False)
df["Working_day"] = [int(i) for i in df["Working_day"]]

#delete the day variable and apply one hot encoding to the categorical variables
df2= df.drop("Day", axis=1)
df2 = pd.get_dummies(df2, columns=['Hour',"Weekday"])

#set random seed to obtain constant results
np.random.seed(42)
tf.random.set_seed(42)

#Split in train-test
df2= df.drop("Day", axis=1)
df2 = pd.get_dummies(df2, columns=['Hour',"Weekday"])
X = df2.drop('energy_used', axis=1)
y = df2['energy_used']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale the input data using min-max scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#create the model in keras
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
#configurate the loss function and the optimizer
model.compile(loss='mean_squared_error', optimizer='adam')
#configurate the early stopping strategy
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
#train the model
history = model.fit(X_train_scaled, y_train, epochs=500,
                    batch_size=128, validation_data=(X_test_scaled, y_test)
                    , callbacks=[early_stop])

#save the model and the scaler
save_model(model, 'energy_estimation.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#test the model
y_predicted = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_predicted)
print(f'RMSE in the test data: {rmse}')
print(f'R^2 in the test data: {r2}')

#generate the real vs predicted visualization
y_real = np.array(y_test.copy())
y_pred= np.array([i[0] for i in y_predicted])
sns.set_theme(style="whitegrid")
plt.scatter(y_test, y_pred, s=5, alpha=0.25, color="#5F88A3")
max_val = max(y_test.max(), y_pred.max())
plt.xlabel('Predicted energy consumption (kWh)')
plt.ylabel("Actual energy consumption (kWh)")
errors = y_test - y_pred
lower, upper = np.percentile(errors, [0.5, 99.5])
plt.fill_between([0, max_val], [lower, lower+max_val], [upper, upper+max_val],alpha=0.2, color='#705452', label='99% prediction interval')
lower, upper = np.percentile(errors, [2.5, 97.5])
plt.fill_between([0, max_val], [lower, lower+max_val], [upper, upper+max_val],alpha=0.2, color='#F071667F', label='95% prediction interval')
plt.plot([0, max_val], [0, max_val], '--', color='#F07166', label='Ideal prediction')
plt.legend()
plt.savefig('realvspredicted.png', dpi=300)
plt.show()


