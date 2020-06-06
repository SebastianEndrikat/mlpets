#!/usr/bin/env python3

# tutorial binary addition
# https://peterroelants.github.io/posts/rnn-implementation-part02/
# except using tensorflow

# learning outcome:
# how to piece layers together with desrired input and ouput dimensions for each
# at first, results wouldnt converge to the target, turns out the activation of 
# the rnn layer didnt fit


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
print('Done loading tf.\n\n')

def binnum2str(bn):
  # input: array of integers
  # turn around and into string
  n=len(bn)
  s=''
  for i in range(1,n+1):
    s+=str(int(bn[-i]))
  return s

def binnum2strFloats(bn):
  # input: array of floats
  # turn around and into string
  n=len(bn)
  s=''
  for i in range(1,n+1):
    s+= '%.2f ' %bn[-i]
  return s



# Create dataset
nsam = 2000  # Number of total samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
nps = 7  # Length of the binary sequence
from rnn1_createData import create_dataset
x,y = create_dataset(nsam, nps)
x,xtest,y,ytest=train_test_split(x,y,train_size=0.6)
ns=x.shape[0] # number of training samples
nstest=xtest.shape[0] # number of test samples
print('x tensor shape: ', x.shape)
print('y tensor shape: ', y.shape)



# Start a model
model = tf.keras.Sequential()
#model.add(layers.Embedding(
#	input_dim=ns, # number of samples?
#	output_dim=2, # dimension of embedding
#	input_length=nps # length of sequence
#	)) # just to set the input dim???
# dense layer from 2 to 2 for each of nps, each one of nps is two long...
model.add(layers.Dense(nps*2, input_shape=(nps,2)))


# add a RNN layer 
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
model.add(layers.SimpleRNN(
    units=3, # state in this layer. output dimension
    activation='tanh', 
    #activation='sigmoid', # shit
    #activation='relu', # almost ok
    #activation='linear', # shit
    use_bias=True, 
    #use_bias=False, 
    bias_initializer='zeros',
    kernel_initializer='glorot_uniform',
    #recurrent_initializer='orthogonal', 
    recurrent_initializer='uniform', 
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
    return_sequences=True, # return all cuz sequence is the resulting number
    return_state=False, go_backwards=False, stateful=False,
    unroll=False
))

model.add(layers.Dense(1, input_shape=(nps,3), 
          activation='sigmoid'
          ))

model.summary() # print


model.compile(
	      #loss=tf.keras.losses.MeanSquaredError(),
              loss=tf.keras.losses.BinaryCrossentropy(),
	      #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              #optimizer='sgd', # stochastic gradient descent
              #optimizer='RMSprop',
              optimizer='adam',
              #metrics=['accuracy']
              metrics=['MeanSquaredError'] # this is just for the print I think
		)


# fit: https://www.tensorflow.org/api_docs/python/tf/keras/Model
model.fit(x,y, 
	#batch_size=ns, 
	batch_size=10,
	#batch_size=int(np.floor(ns/100.)), 
	epochs=20)

############### testing
res=model.predict_on_batch(x)
print('Result shape is ', res.shape)
#for i in range(ns):
for i in range(20):
  resi=res[i,:,0]
  pred=binnum2str(np.round(resi)) # one 7-digit number as str
  targ=binnum2str(y[i,:,0])
  if pred==targ: verdict='Good!'
  else: verdict='Wrong.'
  print('Target=['+targ+'], guess=['+pred+'] '+ verdict+' res=[ '+binnum2strFloats(resi)+']')

ngood=np.sum(np.sum(np.round(res[:,:,0])==y[:,:,0],axis=1)==nps);
print('Testing on training data: %i/%i=%f' %(ngood,ns,float(ngood)/ns))


res=model.predict_on_batch(xtest)
ngood=np.sum(np.sum(np.round(res[:,:,0])==ytest[:,:,0],axis=1)==nps);
print('Testing on test data: %i/%i=%f' %(ngood,nstest,float(ngood)/nstest))



