import sklearn as sc
import random
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing (NEED TF 2.1)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
'''
import kerastuner as kt
'''
from keras import backend as K
from numpy import mean
from numpy import std
import matplotlib.cm as cmx
from matplotlib.colors import LogNorm
import timeit

from numpy.random import seed # keras seed fixing import tensorflow as tf
tf.random.set_seed(42)# tensorflow seed fixing


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
      
masses  = pd.read_csv('masses.txt', sep =',', header = None)
masses.drop(1, axis ='columns', inplace=True)
Min_max_scaler = preprocessing.MinMaxScaler()
masses=Min_max_scaler.fit_transform(masses)
masses = pd.DataFrame(masses)

#collect data from file containing the whole dataset
raw_dataset = pd.read_csv('Z=0.02.txt', sep =' ', skipinitialspace=True, header = None)
raw_dataset.drop(724, axis ='columns', inplace=True)#need to drop spurious 724th column

column_names = []
for i in range(len(names_dict)):
 column_names.append(names_dict[i])
raw_dataset.columns = column_names


plot_predictions = pd.DataFrame()

answers = pd.DataFrame()

#Double Loop. Looping over z will move through the masses, and over q will move through Hcontent

for M in range(0,1,1):
 for H in range(0,6,1):
    dataset_2 = raw_dataset[['mass' ,'Hcontent', 'Dynamical timescale (yr)']]
    dataset_2 = dataset_2.iloc[81 * M :81 * (M+1), :]
    dataset_2['radius2'] = raw_dataset['Dynamical timescale (yr)']
    #dataset_2 = dataset_2.loc[dataset_2['mass'] == 1]
    print(dataset_2)
    
    #SPLIT Data into 6 section via Hcontent. Can vary the section/overlap
    
    if H == 0:
         dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <=  0.001]
    if H == 1:
         dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <=  0.07 ]
         dataset_2 = dataset_2.loc[0.0001 <= dataset_2['Hcontent'] ]
    if H == 2:
         dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <=  0.15 ]
         dataset_2 = dataset_2.loc[0.07 <= dataset_2['Hcontent']]
    if H == 3:
         dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <=  0.35 ]
         dataset_2 = dataset_2.loc[0.11 <= dataset_2['Hcontent']]
    if H == 4:
         dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <= 0.63]
         dataset_2 = dataset_2.loc[0.31 <= dataset_2['Hcontent']]
    if H == 5:
         dataset_2 = dataset_2.loc[0.6 <= dataset_2['Hcontent']]

    print(dataset_2)
    
    dataset_2['Dynamical timescale (yr)'] = np.log(dataset_2['Dynamical timescale (yr)'])
    dataset_2['Hcontent'] = np.log(dataset_2['Hcontent'])
    dataset_2['radius2'] = np.log(dataset_2['radius2'])

    
    print(dataset_2)

    # NORMALISATION - chosen over Standardisation as data (H_content vs Radius) not Gaussian.
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset=min_max_scaler.fit_transform(dataset_2)
    
    #convert back to Dataframe for sampling
    dataset = pd.DataFrame(dataset)
    dataset.columns = ['mass' ,'Hcontent', 'radius', 'radius2']
    dataset = dataset[['mass' ,'Hcontent', 'radius']]

    #shuffle data
    train_dataset = dataset.sample(frac=1, random_state=5)

    sample_weights = []
    for i in range(len(train_dataset)):
     sample_weights.append(float(1))
    sample_weights = np.array(sample_weights)
 
    #Training Data
    train_MH_minmax = train_dataset.iloc[:,0:2]
    train_minmax = train_dataset.iloc[:, 2]


    #convert back into numpy array for input into model
    train_MH_minmax = train_MH_minmax.to_numpy()
    train_minmax = train_minmax.to_numpy()

    #BUILD model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4000)
    binit = tf.keras.initializers.Constant(value = 0)
    kinit = tf.keras.initializers.glorot_uniform()
    act = tf.keras.layers.ReLU(negative_slope = 0.1)
    u = 1512
    mean_weights = []

    model = tf.keras.Sequential([
     tf.keras.Input(shape=(2,)),
     layers.Dense(units=312, activation = act, kernel_initializer = kinit, bias_initializer =  binit),
     layers.Dense(units=512, activation = act, kernel_initializer = kinit, bias_initializer =  binit),
     layers.Dense(units=1512, activation = act, kernel_initializer = kinit, bias_initializer =  binit),
     layers.Dense(units=1, activation = act),
     ])
 
 
    #Initial Smaller models to adjust weights
    
    for i in range(0,0,1):

     model.load_weights('errors_2_weights' + str(H))
 
     if i < 5:
      model.compile(optimizer='Adam', loss = 'mae', learning_rate=0.1, kernel_initializer = kinit)
      history = model.fit(train_MH_minmax, train_minmax, epochs =10, callbacks=[callback], verbose =1, validation_split=0.1, sample_weight = sample_weights)
      model.save_weights('errors_2_weights' + str(H))
     else:
      model.compile(optimizer='Adam', loss = 'mae', learning_rate=0.000001, kernel_initializer = kinit)
      history = model.fit(train_MH_minmax, train_minmax, epochs =20, callbacks=[callback], verbose =1, validation_split=0.1)
      model.save_weights('errors_2_weights' + str(H))
    
     predicts = model.predict(train_MH_minmax)
 
     #Produce Sample Weights from the errors
 
     abs_error= []
     for j in range(len(train_dataset)):
      X = ((abs(predicts[j] - train_minmax[j]))) * 100
      abs_error.append(X[0])
 
     for k in range(len(abs_error)):
      if abs_error[k] < 0.01:
       sample_weights[k] += abs_error[k] * 0
       sample_weights[k] = sample_weights[k] * 1
      else:
       sample_weights[k] += abs_error[k] ** 0.2
    
     sample_weights = np.array(sample_weights)
     train_dataset['abs_error'] = abs_error
     train_dataset['weights'] = sample_weights
     mean_weights.append(mean(sample_weights))
     
    #Bulk of Learning Process
    model.load_weights('errors_2_weights' + str(H))
    model.compile(optimizer='Adam', loss = 'mae', learning_rate=0.00000001, kernel_initializer = 'random_normal')
    history = model.fit(train_MH_minmax, train_minmax, epochs =
    1000, callbacks=[callback], verbose =1, validation_split=0.1, sample_weight = sample_weights)
    model.save('model' + str(H))

    #Produce DataFrame of the RESULTS
    train_predictions = model.predict(train_MH_minmax)
    mass = []
    for i in range(len(train_MH_minmax[:,1])):
     X = train_dataset.iloc[i,0]
     mass.append(X)
    results = pd.DataFrame()
    results['mass'] = mass
    results['Hcontent'] = train_MH_minmax[:,1]
    results['predictions'] = train_predictions
    results['true values'] = train_minmax
    
    results = min_max_scaler.inverse_transform(results)
    results = pd.DataFrame(results, columns = ['mass', 'Hcontent', 'radii predicts', 'true values'])
    results = results.sort_values('Hcontent', axis = 0)
    print('results')
    print(results)

    #Refine the results removing the overlap in Hcontent
    
    if H == 0:
         results = results.loc[results['Hcontent'] <=  np.log(0.001)]
    if H == 1:
         results = results.loc[results['Hcontent'] <=  np.log(0.06) ]
         results = results.loc[np.log(0.001) < results['Hcontent'] ]
    if H == 2:
         results = results.loc[results['Hcontent'] <=  np.log(0.13) ]
         results = results.loc[np.log(0.06) < results['Hcontent']]
    if H == 3:
         results = results.loc[results['Hcontent'] <=  np.log(0.33) ]
         results = results.loc[np.log(0.13) < results['Hcontent']]
    if H == 4:
         results = results.loc[results['Hcontent'] <= np.log(0.6)]
         results = results.loc[np.log(0.33) < results['Hcontent']]
    if H == 5:
         results = results.loc[np.log(0.6) < results['Hcontent']]
    print(results)

    x1 = np.array([np.zeros(1000), np.linspace(0, 1,1000)])
    x1 = np.transpose(x1)
    y1 = model.predict(x1)
    
    # Inverse scaling done manually.
    for i in range(len(y1)):
      y1[i] = y1[i] * (max(dataset_2.iloc[:,2])- min(dataset_2.iloc[:,2])) + min(dataset_2.iloc[:,2])
      x1[i,1] = x1[i,1] * (max(dataset_2.iloc[:,1]) - min(dataset_2.iloc[:,1])) + min(dataset_2.iloc[:,1])
    
    predictz = pd.DataFrame(x1, columns = ['mass', 'Hcontent'])
    predictz['y1'] = y1
    
    plot_predictions = pd.concat([plot_predictions, predictz])
    print('plot_predictz')
    
    #Add RESULTS to ANSWERS each time the loop advances
    answers = pd.concat([answers , results])
    print('ANSWERS')
    print(answers)
    
    #PLot the results for each Hcontent section
    plt.scatter(results.iloc[:, 1], results.iloc[:,2], label = 'Predictions',s=4)
    plt.scatter(results.iloc[:, 1], results.iloc[:,3], label = 'Data',s=1)
    plt.plot(x1[:, 1], y1 ,color = 'g')
    plt.xlabel('log Hcontent')
    plt.ylabel('log radius')
    plt.title('Mass = 1.0')
    plt.legend()
   # plt.show()
    
   

#Remove Logarithmns

answers['radii predicts'] = np.exp(answers['radii predicts'])
answers['true values'] = np.exp(answers['true values'])
answers['Hcontent'] = np.exp(answers['Hcontent'])
print(answers)
plot_predictions['Hcontent'] = np.exp(plot_predictions['Hcontent'])
plot_predictions['y1'] = np.exp(plot_predictions['y1'])

#Store the model features in the variable 'history'
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Plot the model loss
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.2])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.title('training loss')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()


#PLot the Training Data vs Predictions
for i in range(0,1,1):
    plt.scatter(answers.iloc[:, 1], answers.iloc[:,2], label = 'Predictions',s=4)
    plt.scatter(answers.iloc[:, 1], answers.iloc[:,3], label = 'Data',s=1)
    plt.plot(plot_predictions.iloc[:,1],plot_predictions.iloc[:,2],color = 'k')
    plt.xlabel('Hcontent')
    plt.ylabel(column_names[i+2])
    plt.title('Mass = 1.0')
    plt.legend()
    plt.show()


#generate predictions and plot some predictions. Produce Errors
 
predicts = model.predict(train_MH_minmax)

train_minmax = np.reshape(train_minmax, (len(train_minmax) , 1) )


abs_error2 = abs(answers.iloc[:,2] - answers.iloc[:,3])
rel_error = abs(100 * ((answers.iloc[:,2] - answers.iloc[:,3]) ) / (answers.iloc[:,3]))
 
rel_error_train = []
for i in range(len(train_minmax)):
 if train_minmax[i] != 0:
  J = 100 * (abs(train_minmax[i] - train_predictions[i]) ) / (train_minmax[i])
  rel_error_train.append(J)
 else:
  rel_error_train.append(0)
  
answers['per_error'] = rel_error


#HEATMAP

df_train_MH_minmax = answers.iloc[:,:2]
df_train_MH_minmax['rel_error'] = rel_error
df_train_MH_minmax = df_train_MH_minmax.round(8)
print(df_train_MH_minmax)
df_train_MH_minmax = df_train_MH_minmax.sort_values('mass',axis =0, )
df_train_MH_minmax = df_train_MH_minmax.sort_values('Hcontent',axis =0, )
print(df_train_MH_minmax)


table = df_train_MH_minmax.pivot_table('rel_error','mass' , 'Hcontent')
ax = sns.heatmap(table, cmap = 'viridis', norm = LogNorm())
ax.invert_yaxis()
ax.set_title('Error Heatmap with log colour scale - train,Z=0.02')
ax.collections[0].colorbar.set_label("% error")
print(table)
plt.show()


print(mean(rel_error))
