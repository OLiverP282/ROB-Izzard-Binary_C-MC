import sklearn as sc
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



column_names = ['Mass', 'H_content', 'Radius']
dataset = pd.read_csv('1SM_13C.txt', names = column_names, sep = ' ',skipinitialspace=True,  header = None)
dataset.drop('Mass', axis = 'columns', inplace = True)


train_dataset = dataset.sample(frac=0.8, random_state=10)
test_dataset = dataset.drop(train_dataset.index)

#Seperation of the data sets
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('Radius')
test_labels = test_features.pop('Radius')


#NORMALISATION - chosen over Standardisation as data (H_content vs Radius) not Gaussian.
min_max_scaler = preprocessing.MinMaxScaler()
train_Hcontent_minmax = min_max_scaler.fit_transform(train_features)
test_Hcontent_minmax = min_max_scaler.fit_transform(test_features)
train_Radii_minmax = min_max_scaler.fit_transform(train_labels.values.reshape(-1,1))
test_Radii_minmax = min_max_scaler.fit_transform(test_labels.values.reshape(-1,1))


#BUILD model
def model_builder(hp):
 model = tf.keras.Sequential()
 model.add(tf.keras.Input(shape=(1,))) 
 hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
 model.add(layers.Dense(units=hp_units, activation = 'relu')) #train the number of units
 model.add(layers.Dense(units=hp_units, activation = 'relu'))
 model.add(layers.Dense(units=hp_units, activation = 'relu'))
 model.add(layers.Dense(units=1))
 hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) #some choices of LearningRate to try
 model.compile(optimizer='Adam', loss = 'mae', learning_rate=hp_learning_rate,)
 return model

#searches for ideal hps
tuner = kt.Hyperband(model_builder, objective='val_loss', max_epochs=20) #method of searching the various hp combinations
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) #to prevent overtraining
tuner.search(train_Hcontent_minmax,train_Radii_minmax, epochs=20, validation_split=0.2,verbose = 2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print({best_hps.get('units')})
print({best_hps.get('learning_rate')})

#Run the model on these parameters
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
Radii_model = tuner.hypermodel.build(best_hps)
history = Radii_model.fit(train_Hcontent_minmax, train_Radii_minmax,callbacks=[callback],  epochs = 100, verbose =2 , validation_split=0.2)
Radii_model.summary()



#Plot the errors
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.2])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
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


#test the model
test_Radii_predictions = Radii_model.predict(test_Hcontent_minmax)
plt.scatter(test_Hcontent_minmax, test_Radii_predictions, label = 'Predictions')
plt.scatter(test_Hcontent_minmax, test_Radii_minmax, label = 'Data')
plt.xlabel('H_content')
plt.ylabel('Radius')
plt.legend()
plt.show()

