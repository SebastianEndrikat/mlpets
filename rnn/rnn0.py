#!/usr/bin/env python3
# 2020-05-01
# based on:
# https://peterroelants.github.io/posts/rnn-implementation-part01/

# the model as one recurrent weight and one input weight
# the input is a sequence of ones and zeros
# we want to count the ones


import numpy as np
import matplotlib.pyplot as plt

# Settings
alpha=0.1 # learning rate
deltax=0.01 # initial weight-change amount
deltar=0.01 # initial weight-change amount
np.random.seed(42)

# data
ns=40 # number of sequences
nps=5 # number of elements in one sequence
x=(np.round(np.random.rand(ns*nps)).astype(int)).reshape((ns,nps))
y=np.sum(x,axis=1)

# for this example, weights are each just a 1x1 matrix
wr=0.1 # recurrent weight
wx=0.8 # input-weight
h0=0. # initial value of the 1x1 matrix of neurons
#wx=1. ;wr=1. # cheating to get perfect model
# if I set the initial weights to the same value, they will converge to the perfect solution
# if the initial values are unqeual, the model doesnt find the perfect weights

def fprop(x,h0,wr,wx):
 # forward propagation
 # x is here just one sequence with nps elements
 nps=len(x)
 hs=np.zeros(nps) # all the states the network takes during this sequence
 for ll in range(nps):
  if ll==0: prevh=h0
  else: prevh=hs[ll-1]
  hs[ll]= prevh*wr + x[ll]*wx
 return hs

ne=100
signx0=1.0 # initialize previous sign
signr0=1.0
MSE=np.zeros(ne)
for ee in range(ne): # training epochs
 # run thru the data set and compute the error
 dmsedyhat=0. # initialize gradient of mean squared error w.r.t. yhat
 for ii in range(ns): # for all sample sequences
  hs=fprop(x[ii,:],h0,wr,wx)
  yhat=hs[-1] # prediction is the last state
  #yhat=np.round(hs[-1]) # round to integer, as we're counting integers
  MSE[ee]+=(1./ns)*(yhat-y[ii])**2.
  dmsedyhat+= (1./ns)*2.0*(yhat-y[ii])
 print('epoch=%i MSE=%.8f' %(ee,MSE[ee])) 

 # propagate the error backwards
 dwx=0.; dwr=0.
 # no need to randomize order of samples, because the batchsize==samplesize
 # i.e. we run thru the whole data set before updating weights anyways, no matter the order
 for ii in range(ns): # all samples
  e=dmsedyhat # for the very last state, the error is dMSE/dyhat
  for ll in range(nps-1,-1,-1):
   if ll==0: prevh=h0
   else: prevh=hs[ll-1]
   dwx += (1./ns)*e*x[ii,ll] # change in wx, i.e. dwx= e[ll]*x[ll]
   dwr += (1./ns)*e*prevh # change in wr, i.e. dwr= e[ll]*hs[ll-1]
   e*= wr # prepare error for next step back, e[ll-1] = e*wr
 print('epoch=%i Would change weights: dwr=%.6f, dwx=%.6f' %(ee,dwr,dwx))
 if np.abs(dwx)>0.: # i.e. dont check, always clip weight-changes
  signx=np.sign(dwx)
  dwx = signx*deltax # clip
  if signx != signx0:
   deltax *= 0.5 # sign changed
  else:
   deltax *= 1.2 # sign didnt change
  signx0=signx
 if np.abs(dwr)>0.: # i.e. dont check, always clip weight-changes
  signr=np.sign(dwr)
  dwr = signr*deltar # clip
  if signr != signr0:
   deltar *= 0.5 # sign changed .. if this is larger, result is the same but a bit less stable
  else:
   deltar *= 1.2 # sign didnt change
  signr0=signr
 print('epoch=%i Will change weights:  dwr=%.6f, dwx=%.6f' %(ee,dwr,dwx))
 wx-=alpha*dwx; wr-=alpha*dwr; # update weights

print('Final weights wr=%f, wx=%f' %(wr,wx))
print('Testing on final model:')
for ii in range(ns): # for all sample sequences
 hs=fprop(x[ii,:],h0,wr,wx)
 yhat=np.round(hs[-1]) # round to integer, as we're counting integers
 if yhat==y[ii]:
  res='Good!'
 else:
  res='Miscounted.'
 print('Sample %04i: prediction=%i, true value = %i ' %(ii,yhat,y[ii]) + res)

# plot convergence
plt.figure()
plt.plot(np.arange(ne)+1,MSE,'k-')
plt.xlabel('epochs')
plt.ylabel('mean squared error')
#plt.yscale('log')
plt.title('Final error = %.6f' %(MSE[-1]))
#plt.show()
plt.savefig('rnn0_result.png',dpi=200,bbox_inches='tight')



