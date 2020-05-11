#!/usr/bin/env python3

# same as rnn0.py but using tensor flow instead of writing the code


# takeaways:
# start a model, embedd the input (i.e. layer 0)
# add layers
# activation of last layer has to fit the desired result, obviously
# loss function choices, some may not make sense
# weight initialization makes all the difference once again!
# also batch size has major effects. Smaller looks to be better in this case


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras import layers

print('Done loading tf.\n\n')


# data
ns=40 # number of sequences
nps=5 # number of elements in one sequence
x=(np.round(np.random.rand(ns*nps)).astype(int)).reshape((ns,nps))
y=np.sum(x,axis=1)

# Start a model
model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=nps, output_dim=nps))

# add a RNN layer 
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
model.add(layers.SimpleRNN(
    units=1, # only one state in this layer. output dimension
    #activation='tanh', # cant use tanh because output needs to be greater than 1
    activation='linear', 
    #use_bias=True, 
    use_bias=False, 
    bias_initializer='zeros',
    kernel_initializer='glorot_uniform',
    #recurrent_initializer='orthogonal', 
    recurrent_initializer='uniform', 
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
    return_sequences=False, return_state=False, go_backwards=False, stateful=False,
    unroll=False
    #,input_dim=nps # pass this to auto-build the model
))


model.summary() # print

model.compile(
		loss=tf.keras.losses.MeanSquaredError(),
		#loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer='sgd', # stochastic gradient descent
              #optimizer='RMSprop',
              #metrics=['accuracy']
              metrics=['MeanSquaredError'] # this is just for the print I think
		)


# fit: https://www.tensorflow.org/api_docs/python/tf/keras/Model
model.fit(x,y, 
	#batch_size=ns, 
	batch_size=int(np.floor(ns/6.)), 
	epochs=20)


res=model.predict_on_batch(x)
for i in range(ns):
 pred=np.round(res[i])
 if pred==y[i]: verdict='Good!'
 else: verdict='Wrong.'
 print('Target=%i, guess=%f ' %(y[i],res[i])+ verdict)


