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

#collect data from file containing the whole dataset
dataset = pd.read_csv('Whole.txt', sep = ' ',skipinitialspace=True,  header = None)
dataset.drop(723, axis ='columns', inplace=True) #need to drop spurious 724th column 


#NORMALISATION - chosen over Standardisation as data (H_content vs Radius) not Gaussian.    
min_max_scaler = preprocessing.MinMaxScaler()
dataset=min_max_scaler.fit_transform(dataset)

#convert back to Dataframe for sampling                                                     
dataset = pd.DataFrame(dataset)
train_dataset = dataset.sample(frac=0.8, random_state=8)
test_dataset = dataset.drop(train_dataset.index)
test_dataset = test_dataset.sample(frac = 1) #shuffles the test data also


#Seperation of the data sets                                                                            
train_MH_minmax = train_dataset.iloc[:,0:2]
test_MH_minmax = test_dataset.iloc[:, 0:2]
train_minmax = train_dataset.iloc[:, 2:]
test_minmax = test_dataset.iloc[:, 2:]

#convert back into numpy array for input into model                                         
train_MH_minmax = train_MH_minmax.to_numpy()
test_MH_minmax = test_MH_minmax.to_numpy()
train_minmax = train_minmax.to_numpy()
test_minmax = test_minmax.to_numpy() #these are arrays of shape (x,) which is dangerous but seems fine                                                                     


#BUILD model                                                                                                                          
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(2,)))
#can add as many layers as necessary here
model.add(layers.Dense(units=1000, activation = 'relu'))                   
model.add(layers.Dense(units=1000, activation = 'relu'))
model.add(layers.Dense(units=721))
model.compile(optimizer='Adam', loss = 'mae', learning_rate=0.01)
history = model.fit(train_MH_minmax, train_minmax, epochs = 150,callbacks=[callback], verbose =2 , validation_split=0.2)
model.summary()
model.input_shape


#Store the results in the variable 'history'                                                 
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Plot the errors                                                                                                                                                                                          
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.08])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()


#generate predictions and plot some predictions. Plotting a selection of the 721 against Hcontent
test_predictions = model.predict(test_MH_minmax)
for i in range(0,721, 9):
    plt.scatter(test_MH_minmax[:,1], test_predictions[:,i], label = 'Predictions',s=2)
    plt.scatter(test_MH_minmax[:,1], test_minmax[:,i], label = 'Data',s=2)
    plt.xlabel('Hcontent')
    plt.ylabel('Feature' + str(i))
    plt.legend()
    plt.show()
