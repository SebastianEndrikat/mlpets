#!/usr/bin/env python3


# tensorflow version of the same network as written all by hand


def loadNIST():
    theDir='../digits'
    xraw=np.loadtxt(theDir+'/nist_x_data.csv',delimiter=',')
    yraw=np.loadtxt(theDir+'/nist_y_data.csv')
    
    print('Input data in [%f, %f]' %(np.min(xraw),np.max(xraw)))
    
    nx,nf=xraw.shape # nSample, nFeatures
        
    yraw[yraw==10.]=0.

    y=np.zeros((nx,10),dtype=np.int)
    for i in range(nx):
      y[i,int(yraw[i])]=1. # 1 in this position, 0 elsewhere
        
    return xraw,y
 

import matplotlib.pyplot as plt
import numpy as np
import time, sys
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras import layers

x,y=loadNIST()
nx=len(x) # number of samples
nf=len(x[0]) # number of features per sample

print('Done loading tf.\n\n')

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(
    	units=25,
	input_shape=(nf,), 
	activation='relu', 
	#activation='sigmoid', 
	use_bias=True, kernel_initializer='glorot_uniform',
    	bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    	activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
	))


model.add(tf.keras.layers.Dense(
    	units=10, 
	activation='sigmoid', 
	use_bias=True, kernel_initializer='glorot_uniform',
    	bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
    	activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
	))


model.summary() # print

model.compile(
		loss=tf.keras.losses.MeanSquaredError(),
		#loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              #optimizer='sgd', # stochastic gradient descent
              optimizer='RMSprop',
              #metrics=['accuracy']
              metrics=['MeanSquaredError'] # this is just for the print I think
		)

# optmimzed sgd     and batch size 10: 55.58 percent on training data
# optmimzed RMSprop and batch size 10: 96.04 percent on training data
# same but sigmoid activation in first layer instead of relu: 94.9 percent
# optmimzed RMSprop and batch size 1 : 94.96 percent on training data (relu again). takes long


# fit: https://www.tensorflow.org/api_docs/python/tf/keras/Model
model.fit(x,y, 
	#batch_size=nx, 
	batch_size=20, 
	epochs=20)


# test
res=model.predict_on_batch(x)
preds=np.argmax(res,axis=1)
targs=np.argmax(y,axis=1)
ngood=np.sum(preds==targs)
print('Testing on training data: %i/%i correct, i.e. %.4f percent' %(ngood,nx,100.*ngood/float(nx)))



