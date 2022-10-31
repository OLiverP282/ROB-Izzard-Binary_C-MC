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
from scipy.interpolate import interp1d
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from keras import backend as K
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
import matplotlib.cm as cmx
from matplotlib.colors import LogNorm
import timeit
import scipy
from boltons import iterutils


def data_indexer(y_input):
    N = len(y_input)
    index_store= np.array([])
    for i in range(1,N):
        k = i-int(len(index_store))
        av = np.average(y_input[:k])
        if np.abs(y_input[k]-av) > 0.2 * av:
            index_store = np.append(index_store,i)
            y_input = np.delete(y_input,k)
        else:
            pass
    return index_store.astype(int)



y = np.array([1,1,1,1,1,9,9,9,1,1,5,5,5,5,1,1,1,1,9,9,9,1,1,5,5,5,5,1,1,1,1,9,9,9,1,1,5,5,5,5,1,1,1,1,9,9,9,1,1,5,5,5,5])
x= np.arange(0,len(y),1)
TOTAL_STORE = []
def data_splitter(y_input):
    index_store = data_indexer(y_input)
    C = np.arange(0,len(y_input),1)
    inv_store= np.delete(C,index_store).astype(int)
    TOTAL_STORE.append(inv_store)
    d= y_input[index_store]
    if len(d) == 0:
        return None
    else:
        data_splitter(d)


data_splitter(y)


Index_SEGMENTS= []
def original_index_recover(y_input):
    f = np.arange(0, len(y_input), 1)
    for i in range(len(TOTAL_STORE)):
        Index_SEGMENTS.append(f[TOTAL_STORE[i]])
        f = np.delete(f,TOTAL_STORE[i])

original_index_recover(y)


print(Index_SEGMENTS)
def slice_point_identifier(index_segments):
    slice_point_store = []
    for i in range(len(index_segments)):
        for j in range(len(1,index_segments[i])):
            if index_segments[i][j] - index_segments[i][j-1] != 1:
                slice_point_store.append(index_segments[i][j])
    return slice_point_store


# Use slice points to chop data appropiately
slicepoints= slice_point_identifier(Index_SEGMENTS)


# Converts index points to actual x/values
def index_assigner(Index_SEGMENTS,x,y):
    Y_segments = []
    X_segments = []
    for i in range(len(Index_SEGMENTS)):
        X_segments.append(x[Index_SEGMENTS[i]])
        Y_segments.append(y[Index_SEGMENTS[i]])
    return X_segments, Y_segments



# Experimental not yet functional, attempts to return distinct ready to process slices, without relying on inputing
# into Ed's slices
def sequential_seperator(y_segments):
    gap_point_store = []
    gap_store = []
    d=1
    def arrayreshaper_1(array, N):
        X = 2*int(np.ceil(len(array) / N))-1
        addon = np.zeros(X * N - len(array))
        d = np.append(array, addon)
        d = np.resize(d, (X, N))
        return d
    for i in range(len(y_segments)):
        for j in range(0,len(y_segments[i])-1):
            k = j + d
            c = y_segments[i][k-1] - y_segments[i][k-2]
            if c != 1:
                # y_segments[i] = np.delete(y_segments[i],j)
                d += 1
            else:
                pass
    # Z = arrayreshaper_1(gap_point_store, 2).astype(int)
    # for i in range(int(len(Z)/2)):
    #     c = np.arange(Z[2*i-2][1],Z[2*i-1][1]+1,1)
    #     d = np.zeros((len(c),2))
    #     d[:,0]=Z[i][0]
    #     d[:,1]=c

        # gap_store.append(d)


# sequential_seperator(Y_SEGMENTS)
# print(Y_SEGMENTS)

# Function to split up data manually and return distinct arrays to use.
def manual_data_chopper(x_data,y_data,Div_No):
    if len(x_data) != len(y_data):
        print('ERROR: Please input data of equal length.')
        quit()
    C = len(y_data)
    D= int(np.ceil(C/Div_No))
    def arrayreshaper(array, N):
        X = int(np.ceil(len(array)/N))
        addon = np.zeros(X * N - len(array))
        d = np.append(array, addon)
        d = np.resize(d, (X, N))
        return d
    x_return ,y_return = arrayreshaper(x_data,D), arrayreshaper(y_data,D)
    return np.split(x_return,len(x_return)), np.split(y_return,len(y_return))




def data_seperator(x_input,y_input):
    def datascharacteriser(y_input):
        N = len(y_input)
        y_array = np.array([])
        index_store = np.array([])
        for i in range(1, N):
            av = np.average(y_input[:i])
            if y_input[i] > 2 * av:
                y_array = np.append(y_array, y_input[i])
                index_store = np.append(index_store, i)
            else:
                pass
        y_remainder = np.delete(y_input, index_store.astype(int))
        return y_array, y_remainder
    data_store = np.array([])
    def data_iterator(data_store,y_input):
        y_array, y_remainder, = datascharacteriser(y_input)
        if y_array.size==0:
            return y_remainder, data_store
        else:
            data_store = np.append(data_store,y_array)
            return data_iterator(data_store,y_remainder),data_store
    if len(x_input) != len(y_input):
        print("ERROR: Please enter datasets of equal length.")
        quit()
    print(data_iterator(data_store,y_input))


def array_sorter(x,y):
    c = np.sort(x)
    d = np.zeros((len(y)))
    for i in range(len(c)):
        d[i] = y[np.where(c[i] == x)]
    return c,d

def interpolater(arrayx,arrayy,x):
    inter_func = interp1d(arrayx, arrayy)
    return inter_func(x)

def KSbinarytester(xdata,ydata,model1):
    # Uses a binary search approach to indentify regions of worst match and then returns the corresponding x value
    if len(xdata) != len(ydata):
        print('Error: x data and y data are not the same length')
        return None
    n = len(xdata)
    avtest = np.zeros(n)
    # binary testing method
    if n > 5:
        l = int(n / 2)
        A1 = ydata[l:]
        A2 = ydata[:l]
        G1 = xdata[l:]
        G2 = xdata[:l]
        K1 = scipy.stats.kstest(G1,model1(A1))
        K2 = scipy.stats.kstest(G2,model1(A2))
        if K1 > K2:
            KSbinarytester(G1, A1, model1)
        else:
            KSbinarytester(G2, A2, model1)
            # for small sample sizes less than 5 a deviation from expect mean approach is employed, this is the sensitivity cap which can be varied as needed
    for i in range(n):
            avtest[i] = np.abs(np.average(np.delete(xdata, i)) - np.average(model1(ydata)))
    return ydata[avtest.argmin()]


def sample_seperator(xdata,ydata,xrem,yrem,model1,N):
    # Program to separate out points in sample which don't match with model, for self matching use interpolated initial region of data to act as model
    # uses KSbinarytester to iterate for a specified number of points, the number of iterations should be guided
    # by result from model_fraction_searcher
    if len(ydata) != len(xdata):
        print('Error: x data and y data are not the same length')
        return None
    if len(xrem) < N:
        ix = np.where(xdata == KSbinarytester(ydata, xdata, model1))
        agerem_N, gasrem_N = np.append(xrem, xdata[ix]), np.append(yrem, ydata[ix])
        return sample_seperator(np.delete(xdata, ix), np.delete(ydata, ix), agerem_N, gasrem_N, model1, N)
    elif len(xrem) >= N:
        Arem , Grem  = array_sorter(xrem, yrem)
        return xdata, ydata, Arem, Grem

