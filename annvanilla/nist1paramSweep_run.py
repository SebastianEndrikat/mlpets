#!/usr/bin/env python2

# run a paramter sweep on the nist data set and save performance data

import numpy as np
import time, sys, os
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
    


def batchTest(model,x,y,ll,logFile):
    # run thru the data set and test how good the model is for each output
    
    # nan check
    if np.any(np.isnan(model.forwardProp(x[0]))):
        raise ValueError('Found NAN in model output!')
    
    nx=len(x)
    pred=np.zeros(nx,dtype=np.int)
    for i in range(nx):
        pred[i]=np.argmax(model.forwardProp(x[i]))
    ngood=np.sum(pred==y)
    
    pr=np.zeros(11)
    for i in range(10):
        pr[i]=np.sum(pred==i)/float(np.sum(y==i))
    pr[10]=ngood/float(nx)

    Wmean, Wmedian, Wstd=model.getWeightStats()

    with open (logFile, 'a') as fid:
        fid.write('%i' %(ll+1))
        for i in range(11):
            fid.write('\t%.4f' %pr[i])
        fid.write('\t%.6f' %Wmean)
        fid.write('\t%.6f' %Wmedian)
        fid.write('\t%.6f' %Wstd)
        fid.write('\n')
    return


#%% load
xall,yall=loadNIST()

from sklearn.model_selection import train_test_split
x,xtest,y,ytest=train_test_split(xall,yall,test_size=0.2,random_state=4)


nx=len(x) # number of available training samples
nin=20*20
nout=10


#%% settings    
hiddenLayers=np.array([25])
nt=50 # number of learning passes thru the whole data set


#alphas=np.array([0.05,0.1,0.15,0.2,0.25,0.3]) # relu blows up for alpha=0.4
#batchSizes=np.array([1,5,10,20,50,100]) # sample size has to be divisible by this so far

alphas=np.array([0.1]) # no need to run all, just look at lambda now
batchSizes=np.array([10]) # sample size has to be divisible by this so far
#lambds=np.array([0.0,1e-6,5e-6, 1e-5,5e-5, 1e-4,2.5e-4,5e-4,7.5e-4, 1e-3,2e-3,1e-2])
lambds=np.linspace(0.,0.003,7)


nalpha=len(alphas)
nbatchSize=len(batchSizes)
nlambd=len(lambds)
nruns=nalpha*nbatchSize*nlambd*2.
print('Starting %i runs.' %nruns)


for af in np.array(['sigmoid','relu']):
    if af=='sigmoid':
        activFuns=np.array(['sigmoid'])
    else:
        activFuns=np.array(['relu'])
        
    for alpha in alphas:
        for batchSize in batchSizes:
            for lambd in lambds:
                
                istr=af+'_%.3f_%04i_%.10f' %(alpha,batchSize,lambd)
                logFileTrain='stats/'+istr+'_train.log'
                logFileTest ='stats/'+istr+'_test.log'
                
                if not os.path.exists(logFileTest):
                    print('Starting '+istr); sys.stdout.flush()
                
                    model=ann(nin,nout,hiddenLayers,alpha=alpha,lambd=lambd,activFuns=activFuns)
                    
                    nb=nx/batchSize # number of batches
    
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
                            
                            
                        batchTest(model,x,y,ll,logFileTrain)
                        batchTest(model,xtest,ytest,ll,logFileTest)
                        t1=time.time()
                        print('Done iteration %i after %.2f sec.' %(ll,(t1-t0)))
                
                
                
                
                




