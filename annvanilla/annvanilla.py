#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:48:32 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
"""



import numpy as np
import sys

# TODO
# lists instead of dicts?
# write a training wrapper for blind use without interest in performance details


class ann():
    
    def __init__(self,nin,nout,hiddenLayers,activFuns=None,alpha=0.1,lambd=0.0):
        # len(hiddenLayers) = number of hidden layers
        # elements in hiddenLayers are number of elements in each layer
        # activFuns=np.array(['sigmoid','relu']) ...  strings for each hidden layer
        # input layer doesnt need activation and output layer is always sigmoid
        # if activFuns==None, sigmoid is used for all
        
        nHiddenLayers=len(hiddenLayers)
        self.nLayers=nHiddenLayers+2
        self.layerSize=np.append(np.append(nin,hiddenLayers),nout)
        self.alpha=alpha # learning rate
        self.lambd=lambd # regularization lambda
        
        np.random.seed(42)
        
        # dont actually need to keep Z[0], what would be for the input layer
        self.Z=dict.fromkeys(range(1,self.nLayers)) # outputs prior to activation function
        for i in range(1,self.nLayers):
            self.Z[i]=np.zeros(self.layerSize[i]) # will be overwritten in forwardProp
            
        self.a=dict.fromkeys(range(self.nLayers)) # outputs after activation
        for i in range(self.nLayers):
            self.a[i]=np.zeros(self.layerSize[i]) # will be overwritten in forwardProp
        
        
        self.W=dict.fromkeys(range(self.nLayers-1))
        for i in range(self.nLayers-1): 
            # initialize random weights for all but the output layer
            eps=(6.0/(self.layerSize[i]+self.layerSize[i+1]))**0.5
            self.W[i]=(2.0*eps*np.random.rand(self.layerSize[i+1],self.layerSize[i])) - eps
        
        self.b=dict.fromkeys(range(self.nLayers-1))
        for i in range(self.nLayers-1):
            # initialize biases with random numbers
            self.b[i]=np.random.rand(self.layerSize[i+1])
        
        
        # for batch training 1:
        self.dW=dict.fromkeys(range(self.nLayers-1))
        for i in range(self.nLayers-1): 
            self.dW[i]=np.zeros((self.layerSize[i+1],self.layerSize[i]))
        
        self.db=dict.fromkeys(range(self.nLayers-1))
        for i in range(self.nLayers-1):
            self.db[i]=np.zeros(self.layerSize[i+1])
       
        # assign activation function to layers
        if np.any(activFuns==None):
            activFuns=np.array([])
            for i in range(nHiddenLayers):
                activFuns=np.append(activFuns,'sigmoid') 
        self.activFuns=dict.fromkeys(range(1,self.nLayers))
        self.dactivFuns=dict.fromkeys(range(1,self.nLayers))
        for i in range(1,self.nLayers-1): # over the hidden layers
            if activFuns[i-1]=='sigmoid':
               self.activFuns[i]=self.sigmoid
               self.dactivFuns[i]=self.dsigmoid
            elif activFuns[i-1]=='relu':
               self.activFuns[i]=self.relu
               self.dactivFuns[i]=self.drelu
        self.activFuns[self.nLayers-1]=self.sigmoid # output layer
        self.dactivFuns[self.nLayers-1]=self.dsigmoid

        self.MSEfile='batchTrainMSE.log'
        with open(self.MSEfile,'w') as fid: # wipe file
            fid.write('# Mean squared error is appended below after every eopch.\n') # write header
            fid.write('# Initializing the network wipes any previous log file!\n')
            fid.write('# nLayers = %i\n' %self.nLayers)
            fid.write('# Layer sizes '+ str(self.layerSize) + '\n')
            fid.write('# learning rate = %e\n' %self.alpha)
            fid.write('# regularization = %e\n' %self.lambd)

        return
    
    def sigmoid(self,x):
#        x[x>=0.99*sys.float_info.max]= 0.99*sys.float_info.max # overflow check
#        x[x<=0.99*sys.float_info.min]= 0.99*sys.float_info.min # overflow check
        return 1./(1.+np.exp(-x)) # logistic function
    def dsigmoid(self,x): # analytic derivative
        return np.exp(-x)/(1.+np.exp(-x))**2.
    
    def relu(self,x):
        n=len(x)
        return np.max(np.vstack((np.zeros(n),x)),axis=0) # relu max(0,x)
    def drelu(self,x):
        n=len(x)
        out=np.zeros(n)
        out[x>=0.]=1.
        return out
    
    
    def forwardProp(self,ain):
        self.a[0]=ain
        for i in range(self.nLayers-1):
            self.Z[i+1] = np.matmul(self.W[i],self.a[i]) + self.b[i]
            self.a[i+1] = self.activFuns[i+1](self.Z[i+1])
        return self.a[self.nLayers-1]
    
    def backwardProp(self,enext):
        # pass the error from the output layer to learn from one single sample
        for i in range(self.nLayers-2,-1,-1): # backwards
            e=np.matmul(np.transpose(self.W[i]),enext) # error at this layer
            grad=self.alpha*enext*self.dactivFuns[i+1](self.Z[i+1]) 
            
            # learn!
            # Delta-weights is grad dot a (a should be transposed but no need for numpy)
            # Delta-bias is just the grad
            self.W[i]+= np.outer(grad,self.a[i])
            self.b[i]+= grad 
            
            enext=e # keep for next layer back
        return
    
    
    def batchTrain1(self,x,y):
        # what I need to average across samples is the dW and db
        # x is a dict of integers in a range. Each entry contains a np array of nin inputs
        # y is a dict of integers in a range. Each entry contains a np array of nout targets
        
        nx=len(x) # number of samples passed
        
        # wipe Deltas
        for i in range(self.nLayers-1):
            self.dW[i]*=0. # elementwise
            self.db[i]*=0.
    
        # call forward for every sample and add to dW and db.
        MSE=0. # added 2020-12-12
        for s in range(nx):
            e = self.forwardProp(x[s])-y[s] # here y[s] as already an array
            MSE+=np.sum(e**2.) # sum over all output nodes
            
            # backprop:
            for i in range(self.nLayers-2,-1,-1): # backwards
                grad=e*self.dactivFuns[i+1](self.Z[i+1]) 
                
                # Delta-weights is grad dot a (a should be transposed but no need for numpy)
                # Delta-bias is just the grad
                self.dW[i]+= np.outer(grad,self.a[i])
                self.db[i]+= grad 
                
                e=np.matmul(np.transpose(self.W[i]),e) # error at this layer, keep for next layer back
                
        # divide by nx to finish averaging the Deltas
        for i in range(self.nLayers-1):
            self.dW[i] /= float(nx)
            self.db[i] /= float(nx)
        MSE /= float(nx*self.layerSize[-1])
            
        # regularization
        for i in range(self.nLayers-1):
            # optional 1/m factor is here called 1/nx 
            self.dW[i] += self.lambd*self.W[i]#/float(nx)
            
        # learn!
        for i in range(self.nLayers-1):
            self.W[i] -= self.alpha*self.dW[i]
            self.b[i] -= self.alpha*self.db[i]
        
        
        return MSE
    
    def trainOneEpoch(self,x,y,nbatch):
        # pass a whole data set.
        # this will call batch train nbatch times
        
        nbatch=int(nbatch) # to make sure
        n=len(x)
        order=np.arange(n)
        np.random.shuffle(order) # randomize order of samples
        nPerBatch=int(n/nbatch) # should probably be divisible. TODO: check
        MSE=0.
        for b in range(nbatch):
            sel=np.arange(b*nPerBatch,(b+1)*nPerBatch)
            MSE+=self.batchTrain1([x[order[i]] for i in sel], [y[order[i]] for i in sel])
        MSE /= (nbatch*1.0)
        
        with open(self.MSEfile,'a') as fid:
            fid.write('%e\n' %MSE)
        
        return
    
    def printWeightStats(self):
        Wmean,Wmedian,Wstd=self.getWeightStats()

        print('Weights stats: ')
        print('mean(W)    = %.6f' %Wmean)
        print('median(W)  = %.6f' %Wmedian)
        print('rms(W)     = %.6f' %Wstd)
        
        return

    def getWeightStats(self):
        W=np.array([])
        for i in range(self.nLayers-1):
            W=np.append(W,self.W[i].ravel())
        Wmean=np.mean(W)
        Wmedian=np.median(W)
        Wstd=np.std(W)
        
        return Wmean, Wmedian, Wstd



    

# =============================================================================
# =============================================================================
if __name__ == "__main__":
    print('Executing this as a script...')
    
#    nin=16*16
#    nout=10
#    hiddenLayers=np.array([16,16])
#    
#    model=ann(nin,nout,hiddenLayers)
#    
#    for i in model.W:
#        print(np.min(model.W[i]),np.max(model.W[i]),model.W[1][2,2])
#    
#    aout=model.forwardProp(np.zeros(nin))    
#    eout=np.copy(aout) # should compare to known of course
#    model.backwardProp(eout)
#    
#    for i in model.W:
#        print(np.min(model.W[i]),np.max(model.W[i]),model.W[1][2,2])
        
        
    
    
    
    
    
