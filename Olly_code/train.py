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
from keras import backend as K
from numpy import mean
from numpy import std
import matplotlib.cm as cmx
from matplotlib.colors import LogNorm
import timeit

Train = 0

# masses = pd.read_csv('masses.csv', sep =',', header = None)
# masses.columns = ['Mass']


#collect data from file containing the whole dataset
raw_dataset = pd.read_csv('grid_Z2.00e-02_chebyshev_without_header.dat.txt', sep =' ', skipinitialspace=True, header = None)
raw_dataset.drop(724, axis ='columns', inplace=True)#need to drop spurious 725th column

masses = pd.read_csv('masses.csv', sep=' ', header=None)
masses.columns = ['Mass']

#Choose mass for predicting
desired_mass = 0.34

#This will sort out the scaling of the desired mass. selects the masses above and below and manually scales w.r.t these.
#Bear in mind this is designed for training of two masses at once. Can be adapted accordingly.
mass_difference = desired_mass - masses.iloc[:,0]
mass_below = float(desired_mass -  min([x for x in mass_difference if x >= 0]))

mass_below_index = (masses[masses['Mass'] == mass_below].index).tolist()
mass_below_index = int(mass_below_index[0])

mass_above = float(masses.iloc[mass_below_index + 1])
desired_mass_scaled = (desired_mass - mass_below)/(mass_above - mass_below)


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

column_names = []
for i in range(len(names_dict)):
 column_names.append(names_dict[i])
raw_dataset.columns = column_names

#set some empty dataframes
plot_predictions = pd.DataFrame()
answers = pd.DataFrame()

# decide the number of masses you wish to train. More masses takes longer
if Train == 1:
    M = (np.linspace(0, 2, 1)).astype(int)

# If not training, this makes sure the correct model is selected for predicting.
elif Train == 0:
    M = [mass_below_index]

# Double Loop. Looping over M will move through the masses, and over H will move through Hcontent
for Mdx in M:
    for H in range(0, 6, 1):
        # select the mass, Hcontent and desired feature; for now this is one feature at a time.
        dataset_2 = raw_dataset[['mass', 'Hcontent', 'radius']]
        dataset_2['radius2'] = raw_dataset['radius']  # This just here to aid with rescaling of the predictions

        # selects two masses for training at a time.
        # To change the starting mass, change the zero to begin later in the masses list.
        Mass1 = float(masses.iloc[Mdx + 0])
        Mass2 = float(masses.iloc[Mdx + 1 + 0])
        dataset_2a = dataset_2.loc[dataset_2['mass'] == Mass1]
        dataset_2b = dataset_2.loc[dataset_2['mass'] == Mass2]
        dataset_2 = pd.concat([dataset_2a])

        # SPLIT Data into 6 section via Hcontent. Can vary the section/overlap

        if H == 0:
            dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <= 0.001]
        if H == 1:
            dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <= 0.06]
            dataset_2 = dataset_2.loc[0.001 <= dataset_2['Hcontent']]
        if H == 2:
            dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <= 0.13]
            dataset_2 = dataset_2.loc[0.06 <= dataset_2['Hcontent']]
        if H == 3:
            dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <= 0.33]
            dataset_2 = dataset_2.loc[0.13 <= dataset_2['Hcontent']]
        if H == 4:
            dataset_2 = dataset_2.loc[dataset_2['Hcontent'] <= 0.6]
            dataset_2 = dataset_2.loc[0.33 <= dataset_2['Hcontent']]
        if H == 5:
            dataset_2 = dataset_2.loc[0.6 <= dataset_2['Hcontent']]

        dataset_2['radius'] = np.log(dataset_2['radius'])
        dataset_2['Hcontent'] = np.log(dataset_2['Hcontent'])
        dataset_2['radius2'] = np.log(dataset_2['radius2'])

        # NORMALISATION - chosen over Standardisation as data (H_content vs Radius) not Gaussian.
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset_2)

        # convert back to Dataframe for sampling
        dataset = pd.DataFrame(dataset)
        dataset.columns = ['mass', 'Hcontent', 'radius', 'radius2']
        dataset = dataset[['mass', 'Hcontent', 'radius']]

        # NORMALISATION - chosen over Standardisation as data (H_content vs Radius) not Gaussian.
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset_2)

        # convert back to Dataframe for sampling
        dataset = pd.DataFrame(dataset)
        dataset.columns = ['mass', 'Hcontent', 'radius', 'radius2']
        dataset = dataset[['mass', 'Hcontent', 'radius']]




# Build own model
# Create easier data slicing interface to use
# Replicate Ed's model
#   - try and create self-improving layer mechanism, ed optimized manually,
#   - by having reconfiguring done by program might make optimizing for different
#   - variables easier and avoid manual configuration
#   -

