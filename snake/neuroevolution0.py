#!/usr/bin/env python3
# inspired by  Greer Viau at
# youtube.com/watch?v=zIkBYwdkuTk
# and The Coding Train

import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import errno
from time import gmtime, strftime
import time
from copy import deepcopy


def mkdir(thedir):
    try:
        os.makedirs(thedir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return

def timeStr():
    return strftime("%Y-%m-%d_%H:%M:%S", gmtime())

class Snake():
    
    def __init__(self,layers,activaStrs=None):
        # layers includes input and output layer
        # provide activaStrs strings for all layers except the last or None
        # activation applied to output layer is argmax by default
        
        #TODO
        # if layers is array of two instances of Snake, create W,b,activFuns from the first
        # instead of random and then copy some from the second instance over
        # mutation in a separate function of Snake
        
        self.layerSize=layers
        self.nLayers=len(layers)
        self.activaStrs=activaStrs # save info
        
        
        self.W=[]
        for i in range(self.nLayers-1): 
            # initialize random weights for all but the output layer
            eps=(6.0/(self.layerSize[i]+self.layerSize[i+1]))**0.5
            self.W.append((2.0*eps*np.random.rand(self.layerSize[i+1],self.layerSize[i])) - eps)
        
        self.b=[]
        for i in range(self.nLayers-1):
            # initialize biases with random numbers
            self.b.append(np.random.rand(self.layerSize[i+1]))
        
        # assign activation function to layers
        if np.any(activaStrs==None): # no strings provided
            activaStrs=[]
            for i in range(self.nLayers-1):
                activaStrs.append('sigmoid') 
        self.activFuns=[] # the functions, not the strings
        for i in range(self.nLayers-1):
            if activaStrs[i]=='sigmoid':
               self.activFuns.append(self.sigmoid)
            elif activaStrs[i]=='relu':
               self.activFuns.append(self.relu)
            elif activaStrs[i]=='tanh':
               self.activFuns.append(self.tanh)
        
        return
    
    def sigmoid(self,x):
        return 1./(1.+np.exp(-x)) # logistic function
    def relu(self,x):
        n=len(x)
        return np.max(np.vstack((np.zeros(n),x)),axis=0) # relu max(0,x)
    def tanh(self,x):
        return np.tanh(x)
    
    def decide(self,a):
        # use sensor data and the currecnt state of the brain to decide which direction to go
        # i.e. a forward propagation thru the network
        # retruns integer of the direction. north (0), east (1), south (2) or west (3)
        for i in range(self.nLayers-1):
            a = self.activFuns[i](np.matmul(self.W[i],a) + self.b[i])
        return int(np.argmax(a))
    
    def mutate(self,rate):
        
        for i in range(self.nLayers-1):
            # could create replacement weights randomly 
            # or nudge existing ones
            # or swap some weights randomly
            
            # nudge by random factor in range sp=std(W)
            sp=np.std(self.W[i])
            wnudge=sp*np.random.rand(self.layerSize[i+1],self.layerSize[i]) -sp/2.
            sp=np.std(self.b[i])
            bnudge=sp*np.random.rand(self.layerSize[i+1]) - sp/2.
            newW=np.copy(self.W[i])*wnudge
            newb=np.copy(self.b[i])*bnudge
            
            wmut=np.random.rand(self.layerSize[i+1],self.layerSize[i])
            bmut=np.random.rand(self.layerSize[i+1])
            self.W[i][wmut<rate]=newW[wmut<rate]
            self.b[i][bmut<rate]=newb[bmut<rate]
        
        return
        
    
def mate(s0,s1,mutationRate=0.01):
    # take two instances of Snake() and return a third
    
    new=Snake(s0.layerSize,s0.activaStrs)
    for i in range(new.nLayers-1):
        # take randomly chosen weights from either parent 
        # except where mutated, keep randomly initialized weights from new 
        wmap=np.random.randint(2,size=(new.layerSize[i+1],new.layerSize[i]))
        bmap=np.random.randint(2,size=new.layerSize[i+1])
        wmut=np.random.rand(new.layerSize[i+1],new.layerSize[i])
        bmut=np.random.rand(new.layerSize[i+1])
        wmap[wmut<mutationRate]=2
        bmap[bmut<mutationRate]=2
        new.W[i][wmap==1]=s0.W[i][wmap==1]
        new.W[i][wmap==0]=s1.W[i][wmap==0]
        new.b[i][bmap==1]=s0.b[i][bmap==1]
        new.b[i][bmap==0]=s1.b[i][bmap==0]
    
    return new
    
def fitness(snake,nx,ny,nutrition=100,noFood=False,gui=False,imgDir='gen'):
    # create an environment with one snoke
    # step in time until the snake dies, return score
    
    # snake is one instance of the class Snake
    # nx, ny are the number of fields in each direction
    # nutrition ... how many steps a snake gains by eating one piece of food
    
    if gui:
        mkdir(imgDir)
    if noFood:
        foodx=-10
        foody=-10 # outside of domain
    else:
        foodx=np.random.randint(low=0,high=nx)
        foody=np.random.randint(low=0,high=ny)
    
    sx=-np.ones(nx*ny,dtype=int)*9 # -9 means not part of snake
    sy=-np.ones(nx*ny,dtype=int)*9
    
    # initial position
    sx[0]=np.floor(nx/2.)
    sx[1]=sx[0]
    sy[0]=np.floor(ny/2.)
    sy[1]=sy[0]+1
    direction=2 # initial direction is down
    n=2 # length of snake. i.e. sx[:n] are non-negative locations
    
    sensor=np.zeros(11) 
    ll=0 # time step
    alive=True
    score=0.
    maxlifespan=2*nutrition
    while alive:
        
        if gui:
            saveField(score,nx,ny,sx,sy,foodx,foody,theFile=imgDir+'/%08i.png' %ll)
            
        # sense:
        # self-awareness only the neck:
#        sensor[0:4]=0. # wipe
#        sensor[direction]=1. # true for direction of head-1
        # self-awareness neck and tail:
#        sensor[0]=sx[1]-sx[0] # xdistance to neck
#        sensor[1]=sy[1]-sy[0] # ydistance to neck
#        sensor[2]=sx[n-1]-sx[0] # xdistance to tail
#        sensor[3]=sy[n-1]-sy[0] # ydistance to tail
        # self-awareness is north/east/south/west free or snake?
#        sensor[0]=np.any(np.logical_and((sx[0]+0)==sx, (sy[0]+1)==sy))
#        sensor[1]=np.any(np.logical_and((sx[0]+1)==sx, (sy[0]+0)==sy))
#        sensor[2]=np.any(np.logical_and((sx[0]+0)==sx, (sy[0]-1)==sy))
#        sensor[3]=np.any(np.logical_and((sx[0]-1)==sx, (sy[0]+0)==sy))
        # self-awareness distance to snake in 4 directions
#        distToNorthernBodyParts=sy[np.isin(sy[sy>sy[0]],sy[0])]-sy[0]
        subset=sy[sy>sy[0]]
        sensor[0]=np.min(np.append(      subset[np.isin(subset,sy[0])]-sy[0],ny)) # north
        subset=sx[sx>sx[0]]
        sensor[1]=np.min(np.append(      subset[np.isin(subset,sx[0])]-sx[0],nx)) # east
        subset=sy[sy<sy[0]]
        sensor[2]=np.min(np.append(sy[0]-subset[np.isin(subset,sy[0])],      ny)) # south
        subset=sx[sx<sx[0]]
        sensor[3]=np.min(np.append(sx[0]-subset[np.isin(subset,sx[0])],      nx)) # west
        # wall sensing:
        sensor[4]=ny-1-sy[0] # number of free fields to the north
        sensor[5]=nx-1-sx[0] # number of free fields to the east
        sensor[6]=sy[0] # number of free fields to the south
        sensor[7]=sx[0] # number of free fields to the west
        if noFood:
            sensor[8]=0.
            sensor[9]=0.
        else:
            sensor[8]=foodx-sx[0]
            sensor[9]=foody-sy[0]
        sensor[10]=n # body length
        
        # decide:
        direction=snake.decide(sensor)
        
        # step:
        sx=np.roll(sx,1)
        sy=np.roll(sy,1)
        # replace head with new location:
        if direction==0: # up
            sx[0]=sx[1]
            sy[0]=sy[1]+1
        elif direction==1: # right
            sx[0]=sx[1]+1
            sy[0]=sy[1]
        elif direction==2: # down
            sx[0]=sx[1]
            sy[0]=sy[1]-1
        elif direction==3: # left
            sx[0]=sx[1]-1
            sy[0]=sy[1]
            
        # eat food or move tail
        if sx[0]==foodx and sy[0]==foody: # ate
            n+=1 # increment snake length
            score+=1.0
            maxlifespan+=nutrition
            # check if completed the game
            if n==int(np.floor(0.8*nx*ny)): # set some max snake length relative to domain size
                alive=False
                causeOfDeath=3 # won!
                score+=10. # big reward for winning
            # create new food outside of snake 
            # TODO without guessing where the snake is 
            while np.any(np.logical_and(np.isin(sx,foodx),np.isin(sy,foody))): # food is in snake
                foodx=np.random.randint(low=0,high=nx)
                foody=np.random.randint(low=0,high=ny)
            
        else: #did not eat
            sx[n]=-9
            sy[n]=-9 # no longer part of the snake
            
            
        # check its not walked into a wall:
        if sx[0]==-1 or sy[0]==-1 or sx[0]==nx or sy[0]==ny:
            alive=False
            causeOfDeath=0
            
        # periodic domain:
#        if sx[0]==-1: sx[0]=nx-1
#        if sy[0]==-1: sy[0]=ny-1
#        if sx[0]==nx: sx[0]=0
#        if sy[0]==ny: sy[0]=0
        
        # check its not walked into itself:
        for i in range(2,n):
            if sx[0]==sx[i] and sy[0]==sy[i]:
                alive=False
                causeOfDeath=1 # self bite
        
        # fractional score as life time award 
        if alive:
            score += 0.005/nutrition # much less than food to reward direct path to food
        
        ll +=1 # end of time step
        if ll==maxlifespan:
            alive=False
            causeOfDeath=2 # starvation
    
    if gui:
        saveScoreImg(score,nx,ny,theFile=imgDir+'/%08i.png' %(ll))
        
    return score,causeOfDeath


    








def imgSetup(nx,ny):
    mpl.rcParams['figure.subplot.left'] = 0.01
    mpl.rcParams['figure.subplot.right'] = 0.99
    mpl.rcParams['figure.subplot.bottom'] = 0.01
    mpl.rcParams['figure.subplot.top'] = 0.96
    
    textwidth=3. 
    figWidth=1.0*textwidth
    figHeight = 1.0*figWidth
    f,ax=plt.subplots(1,1,figsize=(figWidth,figHeight))
    thisax=ax
    thisax.axis('off')
    
    
    thisax.plot([-0.5, nx-1+0.5, nx-1+0.5, -0.5, -0.5],
                [-0.5, -0.5, ny-1+0.5, ny-1+0.5, -0.5],'k-')
    return thisax
    
def saveField(score,nx,ny,sx,sy,foodx,foody,theFile):
    thisax=imgSetup(nx,ny)
    if foodx>=0 and foody>=0.:
        thisax.plot(foodx,foody,'rX')
    n=np.sum(sx>-9)
    color=(0.5,0.5,0.5)
    for i in range(n):
        thisax.fill([sx[i]-0.5, sx[i]+0.5, sx[i]+0.5, sx[i]-0.5, sx[i]-0.5],
                    [sy[i]-0.5, sy[i]-0.5, sy[i]+0.5, sy[i]+0.5, sy[i]-0.5],color=color)
        color=(0.7,0.7,0.7)
    thisax.text(0.01,1.,'Score = %.6f' %score,ha='left',va='top',transform=thisax.transAxes)
    thisax.text(0.99,1., theFile.split('/')[0],ha='right',va='top',transform=thisax.transAxes)
    plt.savefig(theFile,dpi=300)
    plt.close()
    return
def saveScoreImg(score,nx,ny,theFile):
    thisax=imgSetup(nx,ny)
    thisax.text(0.5,0.5,'Final Score\n%.6f' %score,ha='center',va='center',transform=thisax.transAxes)
    thisax.text(0.99,1., theFile.split('/')[0],ha='right',va='top',transform=thisax.transAxes)
    plt.savefig(theFile,dpi=300)
    plt.close()
    return
    

def trainMating(nx,ny,popsize,ngen,mutationRate,hiddenLayers,activations,nutrition,baseDir,plotFreq=9e9):
    
    mkdir(baseDir)
    # network input and output layers
    layers=np.append(np.append(11,hiddenLayers),4)
    
    # build initial population
    pop=[]
    for i in range(popsize):
        pop.append(Snake(layers,activaStrs=activations))
    
    popnew=np.copy(pop) # initialize
    for gen in range(ngen):
        
        # get scores
        scores=np.zeros(popsize)
        for i in range(popsize):
            scores[i],cod=fitness(pop[i],nx,ny,nutrition,gui=False,imgDir='')**2. # SQUARED
#        scores[scores<np.median(scores)]=0. # chuck out lower half of the population
        p=scores/np.sum(scores) # probability of each snake mating
        
        # build mating pool
        inda=np.random.choice(popsize,size=popsize,p=p) # indeces of parents a
        indb=np.random.choice(popsize,size=popsize,p=p) # indeces of parents b
        
        # create new gen by drawing from pool
        for i in range(popsize):
            popnew[i]=mate(pop[inda[i]], pop[indb[i]], mutationRate)
            
        # plot highest score of each gen
        winner=np.argmax(scores)
        print(timeStr()+' Gen=%i, highscore=%f, median score=%f, mean score=%f' %(gen,scores[winner], np.median(scores), np.mean(scores))); sys.stdout.flush()
        writeLog(baseDir+'.log',gen,scores,winner)
        if np.mod(gen,plotFreq)==0.: # let the best snake run again in a new scenario
            fitness(pop[winner],nx,ny,nutrition,gui=True,imgDir=baseDir+'/gen%06i' %gen)
            vFileName='gen%06i' %gen
            with open(baseDir+'/gen%06i/makeVideo.sh' %gen, 'w') as fid:
                fid.write('#!/bin/bash \nffmpeg -framerate 2 -i %08d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+vFileName+'.mp4 \n')
        
#        pop=popnew
        pop=np.copy(popnew)
    
    return

def trainReplace(nx,ny,popsize,ngen,mutationRate,
                 hiddenLayers,activations,nutrition,
                 baseDir,plotFreq=9e9,pop=None):
    # replace low scoring individuals by mutations of high scoring ones
    
    mkdir(baseDir)
    # network input and output layers
    layers=np.append(np.append(11,hiddenLayers),4)
    
    # build initial population
    if np.any(pop==None):
        pop=[]
        for i in range(popsize):
            pop.append(Snake(layers,activaStrs=activations))
    scores=np.zeros(popsize)
    reps=10
    cod=np.zeros((popsize,reps))
    
    for gen in range(ngen):
        
        # get scores
        for i in range(popsize):
            if scores[i]==0.:
                for run in range(reps):
                    sco,cod[i,run]= fitness(pop[i],nx,ny,nutrition,gui=False,imgDir='')
                    scores[i] += sco/float(reps)
        ind=np.argsort(scores)[::-1] # indeces of individuals, starting with highest score
        
        # plot highest score of each gen
        winner=ind[0]
        print(timeStr()+' Gen=%i, highscore=%f, median score=%f, mean score=%f' %(gen,scores[winner], np.median(scores), np.mean(scores))); sys.stdout.flush()
        writeLog(baseDir+'.log',gen,scores,winner,cod)
        if np.mod(gen,plotFreq)==0.: # let the best snake run again in a new scenario
            fitness(pop[winner],nx,ny,nutrition,gui=True,imgDir=baseDir+'/gen%06i' %gen)
            vFileName='gen%06i' %gen
            with open(baseDir+'/gen%06i/makeVideo.sh' %gen, 'w') as fid:
                fid.write('#!/bin/bash \nffmpeg -framerate 2 -i %08d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+vFileName+'.mp4 \n')
        
        # replace loser half of population
        for i in range(int(np.floor(popsize/2.))):
            iwinner=ind[i]
            iloser=ind[popsize-1-i]
            pop[iloser]=deepcopy(pop[iwinner])
            pop[iloser].mutate(rate=mutationRate)
            scores[iloser]=0. # will need to re evaluate
    
    return pop

def trainReplaceNoFood(nx,ny,popsize,ngen,mutationRate,
                       hiddenLayers,activations,nutrition,
                       baseDir,plotFreq=9e9,pop=None):
    # replace low scoring individuals by mutations of high scoring ones
    # dont train to find food, only to avoid walls
    # TODO no need to rescore unchanged snakes
    
    mkdir(baseDir)
    # network input and output layers
    layers=np.append(np.append(11,hiddenLayers),4)
    
    # build initial population
    if np.any(pop==None):
        pop=[]
        for i in range(popsize):
            pop.append(Snake(layers,activaStrs=activations))
    
    for gen in range(ngen):
        
        # get scores
        scores=np.zeros(popsize)
        for i in range(popsize):
            scores[i],cod = fitness(pop[i],nx,ny,nutrition,noFood=True,gui=False,imgDir='')
        ind=np.argsort(scores)[::-1] # indeces of individuals, starting with highest score
        
        # replace loser half of population
        for i in range(int(np.floor(popsize/2.))):
            iwinner=ind[i]
            iloser=ind[popsize-1-i]
            pop[iloser]=deepcopy(pop[iwinner])
            pop[iloser].mutate(rate=mutationRate)
            
        
        # plot highest score of each gen
        winner=ind[0]
        print(timeStr()+' Gen=%i, highscore=%f, median score=%f, mean score=%f' %(gen,scores[winner], np.median(scores), np.mean(scores))); sys.stdout.flush()
        writeLog(baseDir+'.log',gen,scores,winner)
        if np.mod(gen,plotFreq)==0.: # let the best snake run again in a new scenario
            fitness(pop[winner],nx,ny,nutrition,noFood=True,gui=True,imgDir=baseDir+'/gen%06i' %gen)
            vFileName='gen%06i' %gen
            with open(baseDir+'/gen%06i/makeVideo.sh' %gen, 'w') as fid:
                fid.write('#!/bin/bash \nffmpeg -framerate 2 -i %08d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+vFileName+'.mp4 \n')
        
    
    return pop

def writeLog(theFile,gen,scores,winner,cod=None):
    if np.any(cod!=None):
        n=np.prod(cod.size)
        wall=np.sum(cod==0.)/float(n) # this fraction ran into a wall
        bite=np.sum(cod==1.)/float(n) # this fraction bit itself
        food=np.sum(cod==2.)/float(n) # this fraction ran out of steps
        win =np.sum(cod==3.)/float(n) # this fraction ran out of steps
    else:
        wall=0. # not available
        bite=0.
        food=0.
        win=0.
    with open(theFile,'a') as fid:
        fid.write('%04i\t%.8f\t%.8f\t%.8f\t%i\t%.8f\t%.8f\t%.8f\t%.8f\n'
                     %(gen,scores[winner], np.median(scores), np.mean(scores),time.time(),wall,bite,food,win))
    return


nx=20
ny=20
popsize=1000
hiddenLayers=np.array([20,20,20])
#hiddenLayers=np.array([20]) # one hidden layer doesnt work at all! high score 1.21 after 300 gens

#pop=trainReplaceNoFood(nx,ny,
#      popsize,
#      ngen=101,
#      mutationRate=0.1,
#      hiddenLayers=hiddenLayers,
#      activations=np.array(['relu','relu','relu']),
##      activations=np.array(['sigmoid','sigmoid','sigmoid']),
##      activations=np.array(['tanh','tanh','tanh']),
#      nutrition=50, # steps gained per food. should correspond to map size
##      baseDir='/home/sebastian/tmp/snake/run3/noFood',
#      baseDir='run3/noFood',
#      plotFreq=20)

pop=trainReplace(nx,ny,
      popsize,
      ngen=1001,
      mutationRate=0.1,
      hiddenLayers=hiddenLayers,
      activations=np.array(['relu','relu','relu','relu']),
#      activations=np.array(['sigmoid','sigmoid','sigmoid']),
#      activations=np.array(['tanh','tanh','tanh']),
      nutrition=50, # steps gained per food. should correspond to map size
#      baseDir='/home/sebastian/tmp/snake/oneLayer',
      baseDir='run5',
      plotFreq=20, pop=None)



