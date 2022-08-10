import sklearn as sc
import random
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
import kerastuner as kt
from keras import backend as K
from numpy import mean
from numpy import std
import matplotlib.cm as cmx
from matplotlib.colors import LogNorm
import timeit

from numpy.random import seed # keras seed fixing import tensorflow as tf
tf.random.set_seed(42)# tensorflow seed fixing

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

names_dict = {0: 'mass', 1: 'Hcontent', 2: 'radius', 3:'luminosity', 4: 'model age', 5: 'rCore mass', 6: 'rCore radius', 7: 'rCore mass with overshooting', 8: 'rCore radius with overshooting', 9: 'rEnvelope mass', 10: 'rEnvelope radius', 11: 'relative envelope mass from simplified Ledoux', 12:'relative envelope radius from simplified Ledoux', 13: 'apsidal constant', 14 : 'tidal E2 from Zahn', 15: '''tidal E2 from Zahn's lambda''', 16: 'Kelvin-Helmholtz timescale(yr)', 17: 'Dynamical timescale (yr)', 18: 'Nuclear timescale (yr)', 19: ' mean molecular weight at central mesh point', 20: 'mean molecular weight average through star', 21: 'dXcore/dt (yr^-1)', 22:'d2Xcore/dt2 (yr^-2)' }
for i in range(23, 724, 1):
 if i < 123:
  names_dict[i] = 'temperature' + str(i - 22)
 elif i < 223:
    names_dict[i] = 'density' + str(i - 122)
 elif i < 323:
       names_dict[i] = 'total pressure' + str(i - 222)
 elif i < 423:
    names_dict[i] = 'gas pressure' + str(i - 322)
 elif i < 523:
       names_dict[i] = 'radius' + str(i - 422)
 elif i < 623:
    names_dict[i] = 'adiabatic Gamma' + str(i - 522)
 elif i < 723:
       names_dict[i] = 'pressure scale height' + str(i - 622)
 else:
      names_dict[i] = 'New'

raw_dataset = pd.read_csv('whole.txt', sep = ' ',skipinitialspace=True,  header = None)
raw_dataset.drop(724, axis ='columns', inplace=True)
column_names = []
for i in range(len(names_dict)):
 column_names.append(names_dict[i])
raw_dataset.columns = column_names

#Choose Feature and Mass to predict
dataset = raw_dataset[['mass', 'Hcontent', 'Dynamical timescale (yr)']]
dataset_MH = raw_dataset[['mass', 'Hcontent']]
dataset_MH = dataset_MH.iloc[:81*1,:]

masses  = pd.read_csv('masses.txt', sep = ',', header = None,)
masses.drop(1, axis ='columns', inplace=True)
Hcontent = pd.DataFrame(np.linspace(0.0000000001,0.6985,100))

#Produce a large dataset for custom interpolation
dataset_pred = pd.DataFrame()

for i in range(len(masses)):
 for j in range(100):
  W = {'mass' : masses.iloc[i,0], 'Hcontent' : Hcontent.iloc[j,0]}
  Q = pd.DataFrame(data = W, index = [0])
  dataset_pred = dataset_pred.append(Q)
print(dataset_pred)

dataset_pred = dataset_pred.loc[dataset_pred['mass'] == 0.32]
#dataset = dataset.loc[dataset['mass'] == 1]
dataset = dataset.iloc[0 :81 * 1, :]
print(dataset_pred)


#Split via Hcontent, each need seperate scaling and modelling
for q in range(0,6,1):
    if q == 0:
         X0 = dataset_pred.loc[dataset_pred['Hcontent'] <=  0.001]
         Y0 = dataset.loc[dataset['Hcontent'] <=  0.001]
         Y0 = Y0.iloc[:,2]
         Y0 = Y0.values.reshape(-1,1)
    if q == 1:
         X1 = dataset_pred.loc[dataset_pred['Hcontent'] <= 0.06 ]
         X1 = X1.loc[0.001 < X1['Hcontent']]
         Y1 = dataset.loc[dataset['Hcontent'] <= 0.06 ]
         Y1 = Y1.loc[0.001 < Y1['Hcontent']]
         Y1 = Y1.iloc[:,2]
         Y1 = Y1.values.reshape(-1,1)
    if q == 2:
         X2 = dataset_pred.loc[dataset_pred['Hcontent'] <=  0.13 ]
         X2 = X2.loc[0.06 < X2['Hcontent']]
         Y2 = dataset.loc[dataset['Hcontent'] <=  0.13 ]
         Y2 = Y2.loc[0.06 < Y2['Hcontent']]
         Y2 = Y2.iloc[:,2]
         Y2 = Y2.values.reshape(-1,1)
    if q == 3:
         X3 = dataset_pred.loc[dataset_pred['Hcontent'] <=  0.33 ]
         X3 = X3.loc[0.13 < X3['Hcontent']]
         Y3 = dataset.loc[dataset['Hcontent'] <=  0.33 ]
         Y3 = Y3.loc[0.13 < Y3['Hcontent']]
         Y3 = Y3.iloc[:,2]
         Y3 = Y3.values.reshape(-1,1)
    if q == 4:
         X4 = dataset_pred.loc[dataset_pred['Hcontent'] <= 0.6]
         X4 = X4.loc[0.33 < X4['Hcontent']]
         Y4 = dataset.loc[dataset['Hcontent'] <= 0.6]
         Y4 = Y4.loc[0.33 < Y4['Hcontent']]
         Y4 = Y4.iloc[:,2]
         Y4 = Y4.values.reshape(-1,1)
    if q == 5:
         X5 = dataset_pred.loc[0.6 < dataset_pred['Hcontent']]
         Y5 = dataset.loc[0.6 < dataset['Hcontent']]
         Y5 = Y5.iloc[:,2]
         Y5 = Y5.values.reshape(-1,1)

X0['Hcontent'] = np.log(X0['Hcontent'])
X1['Hcontent'] = np.log(X1['Hcontent'])
X2['Hcontent'] = np.log(X2['Hcontent'])
X3['Hcontent'] = np.log(X3['Hcontent'])
X4['Hcontent'] = np.log(X4['Hcontent'])
X5['Hcontent'] = np.log(X5['Hcontent'])

min_max_scaler = preprocessing.MinMaxScaler()

#Scale each of the input Hcontent groups
print(X0)
min_max_scaler_0 = preprocessing.MinMaxScaler()
X0 = min_max_scaler_0.fit_transform(X0)
min_max_scaler_1 = preprocessing.MinMaxScaler()
X1 = min_max_scaler_1.fit_transform(X1)
min_max_scaler_2 = preprocessing.MinMaxScaler()
X2 = min_max_scaler_2.fit_transform(X2)
min_max_scaler_3 = preprocessing.MinMaxScaler()
X3 = min_max_scaler_3.fit_transform(X3)
min_max_scaler_4 = preprocessing.MinMaxScaler()
X4 = min_max_scaler_4.fit_transform(X4)
min_max_scaler_5 = preprocessing.MinMaxScaler()
X5 = min_max_scaler_5.fit_transform(X5)

#Scale the respective feature in order to unscale after

min_max_scaler_6 = preprocessing.MinMaxScaler()
Y0 = min_max_scaler_6.fit_transform(Y0)
min_max_scaler_7 = preprocessing.MinMaxScaler()
Y1 = min_max_scaler_7.fit_transform(Y1)
min_max_scaler_8 = preprocessing.MinMaxScaler()
Y2 = min_max_scaler_8.fit_transform(Y2)
min_max_scaler_9 = preprocessing.MinMaxScaler()
Y3 = min_max_scaler_9.fit_transform(Y3)
min_max_scaler_10 = preprocessing.MinMaxScaler()
Y4 = min_max_scaler_10.fit_transform(Y4)
min_max_scaler_11 = preprocessing.MinMaxScaler()
Y5 = min_max_scaler_11.fit_transform(Y5)

starttime = timeit.default_timer()

#Predict and Unscale

model = tf.keras.models.load_model('model0')
X0_prediction = model.predict(X0)
X0_prediction = min_max_scaler_6.inverse_transform(X0_prediction)
model = tf.keras.models.load_model('model1')
X1_prediction = model.predict(X1)
X1_prediction = min_max_scaler_7.inverse_transform(X1_prediction)
model = tf.keras.models.load_model('model2')
X2_prediction = model.predict(X2)
X2_prediction = min_max_scaler_8.inverse_transform(X2_prediction)
model = tf.keras.models.load_model('model3')
X3_prediction = model.predict(X3)
X3_prediction = min_max_scaler_9.inverse_transform(X3_prediction)
model = tf.keras.models.load_model('model4')
X4_prediction = model.predict(X4)
X4_prediction = min_max_scaler_10.inverse_transform(X4_prediction)
model = tf.keras.models.load_model('model5')
X5_prediction = model.predict(X5)
X5_prediction = min_max_scaler_11.inverse_transform(X5_prediction)

test_predictions = np.concatenate((X0_prediction, X1_prediction, X2_prediction, X3_prediction, X4_prediction, X5_prediction), axis=0)

print("The time difference is :", timeit.default_timer() - starttime)


plt.scatter(dataset_pred.iloc[:,1], test_predictions, s =1)
plt.plot(dataset_pred.iloc[:,1], test_predictions)
plt.scatter(dataset.iloc[:,1], dataset.iloc[:,2], s=3)
plt.show()


