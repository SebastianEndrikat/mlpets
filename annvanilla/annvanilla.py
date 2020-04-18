#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:48:32 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
"""



import numpy as np


# TODO
# lists instead of dicts?
# write a training wrapper for blind use without interest in performance details
# assign one of many activation functions (per layer)


class ann():
    
    def __init__(self,nin,nout,hiddenLayers,alpha=0.1,lambd=0.0):
        # len(hiddenLayers) = number of hidden layers
        # elements in hiddenLayers are number of elements in each layer
        
        nHiddenLayers=len(hiddenLayers)
        self.nLayers=nHiddenLayers+2
        self.layerSize=np.append(np.append(nin,hiddenLayers),nout)
        self.alpha=alpha # learning rate
        self.lambd=lambd # regularization lambda
        
        np.random.seed(42)
        
        # dont actually need to keep Z[0], what would be for the input layer
        self.Z=dict.fromkeys(range(1,self.nLayers)) # outputs prior to activation function
        for i in range(self.nLayers):
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
        
        return
    
    def activFun(self,x):
        return 1./(1.+np.exp(-x)) # logistic function
    def dactivFundx(self,x): # analytic derivative
        return np.exp(-x)/(1.+np.exp(-x))**2.
    
#    def activFun(self,x):
#        n=len(x)
#        return np.max(np.vstack((np.zeros(n),x)),axis=0) # relu max(0,x)
#    def dactivFundx(self,x):
#        n=len(x)
#        out=np.zeros(n)
#        out[x>=0.]=1.
#        return out
    
    
    def forwardProp(self,ain):
        self.a[0]=ain
        for i in range(self.nLayers-1):
            self.Z[i+1] = np.matmul(self.W[i],self.a[i]) + self.b[i]
            self.a[i+1] = self.activFun(self.Z[i+1])
        return self.a[self.nLayers-1]
    
    def backwardProp(self,enext):
        # pass the error from the output layer to learn from one single sample
        for i in range(self.nLayers-2,-1,-1): # backwards
            e=np.matmul(np.transpose(self.W[i]),enext) # error at this layer
            grad=self.alpha*enext*self.dactivFundx(self.Z[i+1]) 
            
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
            
#        regu=0.0
#        for i in range(self.nLayers-1):
#            regu+=np.sum(self.W[i]**2.)
##        regu *= self.lambd/(2.*nx)
#        regu *= self.lambd/2.
    
        # call forward for every sample and add to dW and db.
        for s in range(nx):
            e = self.forwardProp(x[s])-y[s] # here y[s] as already an array
            
            # backprop:
            for i in range(self.nLayers-2,-1,-1): # backwards
                grad=e*self.dactivFundx(self.Z[i+1]) 
                
                # Delta-weights is grad dot a (a should be transposed but no need for numpy)
                # Delta-bias is just the grad
                self.dW[i]+= np.outer(grad,self.a[i])
                self.db[i]+= grad 
                
                e=np.matmul(np.transpose(self.W[i]),e) # error at this layer, keep for next layer back
                
        # divide by nx to finish averaging the Deltas
        for i in range(self.nLayers-1):
            self.dW[i] /= float(nx)
            self.db[i] /= float(nx)
            
        # regularization
        for i in range(self.nLayers-1):
            # optional 1/m factor is here called 1/nx 
            self.dW[i] += self.lambd*self.W[i]#/float(nx)
            
        # learn!
        for i in range(self.nLayers-1):
            self.W[i] -= self.alpha*self.dW[i]
            self.b[i] -= self.alpha*self.db[i]
        
        return
    
    def printWeightStats(self):
        W=np.array([])
        for i in range(self.nLayers-1):
            W=np.append(W,self.W[i].ravel())
        Wmean=np.mean(W)
        Wmedian=np.median(W)
        W2mean=np.mean(W**2.)
        Wstd=np.std(W)
        print('Weights stats: ')
        print('mean(W)    = %.6f' %Wmean)
        print('median(W)  = %.6f' %Wmedian)
        print('mean(W**2) = %.6f' %W2mean)
        print('rms(W)     = %.6f' %Wstd)
        
        return Wmean, Wmedian, W2mean, Wstd



    

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
        
        
    
    
    
    
    
