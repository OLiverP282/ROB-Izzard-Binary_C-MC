import sklearn as sc
import random
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing (NEED TF 2.1)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
import kerastuner as kt


#Collect data and drop the mass column. The data set is the first three columns of the dataset for mass = 1
column_names = ['Mass', 'H_content', 'Radius']
dataset = pd.read_csv('1SM_13C.txt', names = column_names, sep = ' ',skipinitialspace=True,  header = None)
dataset.drop('Mass', axis = 'columns', inplace = True)

#NORMALISATION - chosen over Standardisation as data (H_content vs Radius) not Gaussian. 
min_max_scaler = preprocessing.MinMaxScaler()
dataset=min_max_scaler.fit_transform(dataset)

#convert back to Dataframe for sampling
dataset = pd.DataFrame(dataset, columns = ['H_content', 'Radius'])
train_dataset = dataset.sample(frac=0.8, random_state=8)
test_dataset = dataset.drop(train_dataset.index)

#Separation of the data sets
train_Hcontent_minmax = train_dataset.iloc[:, 0]
test_Hcontent_minmax= test_dataset.iloc[:,0]
train_Radii_minmax = train_dataset.iloc[:,1]
test_Radii_minmax = test_dataset.iloc[:,1]

#convert back into numpy array for input into model
train_Hcontent_minmax = train_Hcontent_minmax.to_numpy()
test_Hcontent_minmax = test_Hcontent_minmax.to_numpy()
train_Radii_minmax = train_Radii_minmax.to_numpy()
test_Radii_minmax = test_Radii_minmax.to_numpy() #these are arrays of shape (x,) which is dangerous but seems fine


#BUILD model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
Radii_model = tf.keras.Sequential()
Radii_model.add(tf.keras.Input(shape=(1,)))
Radii_model.add(layers.Dense(units=200, activation = 'relu'))
Radii_model.add(layers.Dense(units=200, activation = 'relu'))
Radii_model.add(layers.Dense(units=200, activation = 'relu'))
Radii_model.add(layers.Dense(units=200, activation = 'relu'))
Radii_model.add(layers.Dense(units=1))
Radii_model.compile(optimizer='Adam', loss = 'mae', learning_rate=0.01)
history = Radii_model.fit(train_Hcontent_minmax, train_Radii_minmax, epochs = 300,callbacks=[callback], verbose =2 , validation_split=0.2)
Radii_model.summary()
Radii_model.input_shape

#Store the results in the variable 'history'
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Plot the errors
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.2])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Radii]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()

#generate predictions and compare
x = tf.linspace(0.0, 1, 1000)
y = Radii_model.predict(x)

def plot_Radii(x, y):
  plt.scatter(train_Hcontent_minmax,train_Radii_minmax,label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Hcontent')
  plt.ylabel('Radii')
  plt.legend()

plot_Radii(x,y) 

plt.show()



test_Radii_predictions = Radii_model.predict(test_Hcontent_minmax)
plt.scatter(test_Hcontent_minmax, test_Radii_predictions, label = 'Predictions')
plt.scatter(test_Hcontent_minmax, test_Radii_minmax, label = 'Data')
plt.xlabel('H_content')
plt.ylabel('Radius')
plt.legend()
plt.show()

