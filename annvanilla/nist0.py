#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:45:49 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
"""




import numpy as np
import time, sys
from annvanilla import ann

def loadNIST():
    theDir='../digits'
    xraw=np.loadtxt(theDir+'/nist_x_data.csv',delimiter=',')
    yraw=np.loadtxt(theDir+'/nist_y_data.csv')
    
    print('Input data in [%f, %f]' %(np.min(xraw),np.max(xraw)))
    
    nx,nf=xraw.shape # nSample, nFeatures
    
    x=dict.fromkeys(range(nx))
    for i in range(nx):
        x[i]=xraw[i,:]
        
    y=yraw.astype(np.int)
    y[y==10]=0
        
    return x,y
    
def dispImage(x):
    # NIST images are 20x20 pixel
    img=x.reshape((20,20)).transpose()
    import matplotlib.pyplot as plt
    plt.imshow(img)
    
def batchTest(model,x,y,ll):
    # run thru the data set and test how often the model is correct
    
    # nan check
    if np.any(np.isnan(model.forwardProp(x[0]))):
        raise ValueError('Found NAN in model output!')
    
    nx=len(x)
    pred=np.zeros(nx,dtype=np.int)
    for i in range(nx):
        pred[i]=np.argmax(model.forwardProp(x[i]))
    ngood=np.sum(pred==y)
    
    pr=np.zeros(10)
    for i in range(10):
        pr[i]=np.sum(pred==i)/float(np.sum(y==i))
        
    print('pred ratios: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f, total=%.4f' %(pr[0],pr[1],pr[2],pr[3],pr[4],pr[5],pr[6],pr[7],pr[8],pr[9],ngood/float(nx)))
    
    return float(ngood)/float(nx)

def batchTest2(model,x,y,ll):
    # run thru the data set and test how good the model is for each output
    
    # nan check
    if np.any(np.isnan(model.forwardProp(x[0]))):
        raise ValueError('Found NAN in model output!')
    
    nx=len(x)
    pred=np.zeros(nx)
    for i in range(nx):
        pred[i]=model.forwardProp(x[i])[y[i]]
    
    with open ('timeSeriesTests.log', 'a') as fid:
        fid.write('%i' %(ll+1))
        for i in range(10):
            fid.write('\t%.3f' %(np.mean(pred[y==i])))
        fid.write('\n')
    return

    
#%% settings    
hiddenLayers=np.array([100])
activFuns=np.array(['relu'])
#activFuns=np.array(['sigmoid'])
nt=20 # number of learning passes thru the whole data set
alpha=0.1
batchSize=2 # sample size has to be divisible by this so far
lambd=0.001


# things to adjust
# - activation function, type and parameters. how steep...
# - learning rate
# - gotta have enough hidden units. ratio hidden/input>>0.1 maybe? more layers or more elements per layer?
# - batch learning
# - weight initialization makes a huge difference in the beginning
# - regularization, probably important for generalization

#%% load
xall,yall=loadNIST()

from sklearn.model_selection import train_test_split
x,xtest,y,ytest=train_test_split(xall,yall,test_size=0.2,random_state=4)


nx=len(x) # number of available training samples
#dispImage(x[3960])
nin=20*20
nout=10
model=ann(nin,nout,hiddenLayers,alpha=alpha,lambd=lambd,activFuns=activFuns)

## reduce weights to start:
#for i in range(model.nLayers-1):
#    model.W[i] /=100.


#%% train
#printevery=10
#alli=np.arange(nx)
#t0=time.time()
#for ll in range(nt):
#    np.random.shuffle(alli) # random order
#    for i in alli:
#        
#        target=np.zeros(nout); target[y[i]]=1.
#        cost=target-model.forwardProp(x[i])
#        model.backwardProp(cost)
#        
##    print(model.forwardProp(x[0])[0:3])
#        
#    if np.mod(ll+1,printevery)==0:
#        t1=time.time()
#        
##        perf=batchTest(model,x,y)
##        print('Done iteration %i after %.2f sec. Performance=%.6f' %(ll,(t1-t0),perf))
#        
#        batchTest2(model,x,y,ll)
#        print('Done iteration %i after %.2f sec.' %(ll,(t1-t0)))

#%% batch train
nb=nx/batchSize # number of batches

if 0: # train with dictionaries
    targets=dict.fromkeys(range(nx))
    for s in range(nx):
        targets[s]=np.zeros(model.layerSize[-1])
        targets[s][y[s]]=1.
        
    batchx=dict.fromkeys(range(batchSize))
    batchtargets=dict.fromkeys(range(batchSize))
    
    alli=np.arange(nx)
    t0=time.time()
    for ll in range(nt):
        np.random.shuffle(alli) # random order
        for b in range(nb):
            batchi=alli[b*batchSize:(b+1)*batchSize]
            # build sub dicts that only contain samples from this batch
            for s in range(batchSize):
                batchx[s]=x[batchi[s]]
                batchtargets[s]=targets[batchi[s]]
            model.batchTrain1(batchx,batchtargets)
            
        batchTest(model,x,y,ll)
        batchTest(model,xtest,ytest,ll)
    #    model.printWeightStats()
        t1=time.time()
        print('Done iteration %i after %.2f sec.\n' %(ll,(t1-t0)))



#%%

# use lists instead of dicts and then use inbuild function for training to also print MSE
    
xx=[x[i] for i in range(nx)]
yy=[np.zeros(model.layerSize[-1]) for i in range(nx)]
for i in range(nx):
    yy[i][y[i]]=1.
    
t0=time.time()
for ll in range(nt):
    model.trainOneEpoch(xx,yy,nb)


    batchTest(model,x,y,ll)
    batchTest(model,xtest,ytest,ll)
#    model.printWeightStats()
    t1=time.time()
    print('Done iteration %i after %.2f sec.\n' %(ll,(t1-t0)))
