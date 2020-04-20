#!/usr/bin/env python2

# plot the results

import numpy as np
import time, sys, os, glob

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('my.mplstyle')


def loadData():
    fileList=glob.glob('stats/*.log')
    nFiles=len(fileList)
    
    alpha=np.zeros(nFiles)
    batchSize=np.zeros(nFiles)
    lambd=np.zeros(nFiles)
    i=-1
    for theFile in fileList:
        i+=1
        tmp=theFile.split('stats/')[1].split('.log')[0]
        alpha[i]=float(tmp.split('_')[1])
        batchSize[i]=float(tmp.split('_')[2])
        lambd[i]=float(tmp.split('_')[3])
    alpha=np.unique(alpha)
    batchSize=np.unique(batchSize)
    lambd=np.unique(lambd)
    
    nalpha=len(alpha)
    nbatchSize=len(batchSize)
    nlambd=len(lambd)
    
    dat=np.loadtxt(fileList[0])
    nt,ne=dat.shape
    
    res=np.zeros((2,nalpha,nbatchSize,nlambd,2,nt,ne))
    for theFile in fileList:
        tmp=theFile.split('stats/')[1].split('.log')[0]
        s=tmp.split('_')
        
        if s[0]=='sigmoid':
            i=0
        else:
            i=1
        if s[4]=='train':
            m=0
        else:
            m=1
        
        thisalpha=float(s[1])
        j,=np.where(alpha==thisalpha)
        
        thisbatchSize=float(s[2])
        k,=np.where(batchSize==thisbatchSize)
        
        thislambd=float(s[3])
        l,=np.where(lambd==thislambd)
        
        dat=np.loadtxt(theFile)
        if dat.shape[0]==nt: # not still running
            res[i,j,k,l,m,:,:]=dat
        
    return res, alpha, batchSize, lambd





    

def plot0(r,q):

    mpl.rcParams['figure.subplot.wspace'] = 0.2 # width space
    mpl.rcParams['figure.subplot.hspace'] = 0.35 # height space
    mpl.rcParams['figure.subplot.left'] = 0.05
    mpl.rcParams['figure.subplot.right'] = 0.99
    mpl.rcParams['figure.subplot.bottom'] = 0.2
    mpl.rcParams['figure.subplot.top'] = 0.96
    
    textwidth=0.01384*370 # for JFM, 384-14 pt in inches
    figWidth=2.*textwidth
    figHeight = 2*0.8/3.*figWidth
    f,ax=plt.subplots(2,3,figsize=(figWidth,figHeight))
    
    
    if q=='alpha':
        qvals=alpha # the 1D array of available data
        ylabel='learning rate $\\alpha$'
    elif q=='batchSize':
        qvals=batchSize # the 1D array of available data
        ylabel='batch size'
    elif q=='lambd':
        qvals=lambd # the 1D array of available data
        ylabel='regularization $\\lambda$'
    
    its=r[0,0,0,:,0]
    IT,Q=np.meshgrid(its,qvals,indexing='ij')
    
    for thisax in ax.flat:
        thisax.set_xlim([0,50])
        thisax.set_xlabel('epoch')
        thisax.set_ylabel(ylabel)
    
    contourLevelsPerf=np.linspace(0.7,1.0,11)
    contourLevelsRMS=np.linspace(0.0,0.2,11)
    
    i=0 # sigmoid
    m=0 # train
    e=11 # total performance
    plt.set_cmap('Reds')
    thisax=ax[0,0]
    thisax.contourf(IT,Q,r[i,:,m,:,e].T,contourLevelsPerf,extend='both')
    thisax.set_title('Perf. on train. set. Activ.: sigmoid')
    
    
    i=0 # sigmoid
    m=1 # test
    e=11 # total performance
    plt.set_cmap('Reds')
    thisax=ax[0,1]
    thisax.contourf(IT,Q,r[i,:,m,:,e].T,contourLevelsPerf,extend='both')
    thisax.set_title('Perf. on test set. Activ.: sigmoid')
    
    
    i=0 # sigmoid
    m=0 # train/test same weights
    e=14 # rms of weights
    plt.set_cmap('Blues')
    thisax=ax[0,2]
    thisax.contourf(IT,Q,r[i,:,m,:,e].T,contourLevelsRMS,extend='both')
    thisax.set_title('RMS of weights. Activ.: sigmoid')
    
    
    h={} # handles
    
    i=1 # relu
    m=0 # train
    e=11 # total performance
    plt.set_cmap('Reds')
    thisax=ax[1,0]
    h[0]=thisax.contourf(IT,Q,r[i,:,m,:,e].T,contourLevelsPerf,extend='both')
    thisax.set_title('Perf. on train. set. Activ.: ReLU')
    
    i=1 # relu
    m=1 # test
    e=11 # total performance
    plt.set_cmap('Reds')
    thisax=ax[1,1]
    h[1]=thisax.contourf(IT,Q,r[i,:,m,:,e].T,contourLevelsPerf,extend='both')
    thisax.set_title('Perf. on test set. Activ.: ReLU')
    
    
    i=1 # relu
    m=0 # train/test same weights
    e=14 # rms of weights
    plt.set_cmap('Blues')
    thisax=ax[1,2]
    h[2]=thisax.contourf(IT,Q,r[i,:,m,:,e].T,contourLevelsRMS,extend='both')
    thisax.set_title('RMS of weights. Activ.: ReLU')
    
    
    for i in range(3):
        cbaxes = f.add_axes([0.09+0.32*i, 0.0, 0.2, 0.8]) # [left, bottom, width, height]. anchor is 0,0.5
        cbaxes.axis('off')
        cbar = f.colorbar(h[i],ax=cbaxes, extend='both', orientation='horizontal',aspect=15)
#        cbaxes.text(1.5,1.1,'$u^+$',ha='center',va='top')
        cbar.ax.minorticks_off()
    
    
    return f,ax


res,alpha,batchSize,lambd=loadData()
# dimensions:
# i: activFun
# j: alpha
# k: batch size
# l: lambda
# m: train/test
# iterations
# quantities


# set missing data to NAN (relu blows up for some)
for i in range(2):
    for j in range(len(alpha)):
        for k in range(len(batchSize)):
            for l in range(len(lambd)):
                for m in range(2):
                    if res[i,j,k,l,m,-1,0]==0.:
                        res[i,j,k,l,m,:,1:]=np.nan

# =============================================================================
# plot
# =============================================================================

k=2
l=1
f,ax=plot0(res[:,:,k,l,:,:,:],'alpha')
f.text(0.02,0.01,'Constant: batch size = %i, $\\lambda=%.2e$' %(batchSize[k],lambd[l]),ha='left',va='bottom')
ax[1,0].text(8,0.35,'NaN for $\\alpha=0.4$',va='center',ha='left')
plt.savefig('nistsweep_alpha.png',dpi=400)


j=1
l=1
f,ax=plot0(res[:,j,:,l,:,:,:],'batchSize')
f.text(0.02,0.01,'Constant: $\\alpha = %.3f$, $\\lambda=%.2e$' %(alpha[j],lambd[l]),ha='left',va='bottom')
plt.savefig('nistsweep_batchSize.png',dpi=400)

j=1
k=2
f,ax=plot0(res[:,j,k,:,:,:,:],'lambd')
f.text(0.02,0.01,'Constant: batch size = %i, $\\alpha=%.3f$' %(batchSize[k],alpha[j]),ha='left',va='bottom')
for thisax in ax.flat:
    thisax.set_ylim([0,3e-3])
#    thisax.set_yscale('log')
plt.savefig('nistsweep_lambda.png',dpi=400)