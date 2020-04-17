#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:48:32 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
"""



import numpy as np






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
        
        # dont actually need Z[0], eh?
        self.Z=dict.fromkeys(range(self.nLayers)) # outputs prior to activation function
        for i in range(self.nLayers):
            self.Z[i]=np.zeros(self.layerSize[i]) # will be overwritten in forwardProp
            
        self.Zavg=dict.fromkeys(range(self.nLayers)) # used in batchTraining0
        for i in range(self.nLayers):
            self.Zavg[i]=np.zeros(self.layerSize[i])
        
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
        # pass the error from the output layer
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
    
    def batchTrain0(self,x,y):
        # this is BS, doesnt make sense this way
        # x is a dict of integers in a range. Each entry contains a np array of nin inputs
        # y is a dict of integers in a range. Each entry contains a np array of nout targets
        
        nx=len(x) # number of samples passed
        
        # wipe Zavg
        for i in range(self.nLayers):
            self.Zavg[i]*=0. # elementwise
            
        # allocate eavg
        eavg=np.zeros(self.layerSize[-1]) # size of output layer
        
        # call forward for every sample and add Z to Zavg. add to eavg
        for s in range(nx):
            eavg += y[s]-self.forwardProp(x[s]) # here y[s] as already an array
            # Note on convention: often people define cost the other way round and subtract when learning
            for i in range(self.nLayers):
                self.Zavg[i] += self.Z[i]
        
        # divide Zavg and eavg by nx to finish averaging
        eavg /= float(nx)
        for i in range(self.nLayers):
            self.Zavg[i] /= float(nx)
#        print(eavg)
        
        # replace Z and a in self with avg values
        for i in range(self.nLayers):
            self.Z[i]=self.Zavg[i]
        for i in range(self.nLayers-1):
            self.a[i+1] = self.activFun(self.Zavg[i+1])
        
        # call backProp with eavg to learn
        self.backwardProp(eavg)
        
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


def xortest():
    
    x=dict.fromkeys(range(4))
    y=dict.fromkeys(range(4))
    x[0]=np.array([0,1]); y[0]=np.array([1])
    x[1]=np.array([1,0]); y[1]=np.array([1])
    x[2]=np.array([0,0]); y[2]=np.array([0])
    x[3]=np.array([1,1]); y[3]=np.array([0])

    # init
    model=ann(2,1,np.array([2]),alpha=0.1)
#    print model.layerSize
    
    # train
    nt=50000
    logy0=np.zeros(nt) # logging output as I train
    logy1=np.zeros(nt)
    logy2=np.zeros(nt)
    logy3=np.zeros(nt)
    for l in range(nt):
        i=np.random.randint(low=0,high=4) # int in [0,4)
        
        cost=y[i]-model.forwardProp(x[i])
        model.backwardProp(cost)
        
        # test:
        logy0[l]=model.forwardProp(x[0])
        logy1[l]=model.forwardProp(x[1])
        logy2[l]=model.forwardProp(x[2])
        logy3[l]=model.forwardProp(x[3])
        
        
    # test
    for i in range(4):
        print(x[i], y[i], model.forwardProp(x[i]))
#    print(model.b)
        
    import matplotlib.pyplot as plt
    plt.plot(np.arange(nt)+1,logy0,'k-',label='x=[0,1]')
    plt.plot(np.arange(nt)+1,logy1,'r-',label='x=[1,0]')
    plt.plot(np.arange(nt)+1,logy2,'g-',label='x=[0,0]')
    plt.plot(np.arange(nt)+1,logy3,'b-',label='x=[1,1]')
    plt.legend()
    plt.xlabel('learning iteration')
    plt.ylabel('y'); plt.ylim([0,1])
    plt.title('alpha=%f' %model.alpha)
    if 0:
        plt.savefig('xorTest_alpha=%.4f.png' %model.alpha,dpi=400,bbox_inches='tight')
    
    return
    
    

# =============================================================================
# =============================================================================
if __name__ == "__main__":
    print('Executing this as a script...')
    xortest()
    
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
        
        
    
    
    
    
    
