#!/usr/bin/env python3
# inspired by  Greer Viau at
# youtube.com/watch?v=zIkBYwdkuTk
# and The Coding Train

import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import errno
from copy import deepcopy


def mkdir(thedir):
    try:
        os.makedirs(thedir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return

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
            for i in range(self.nLayers):
                activaStrs.append('sigmoid') 
        self.activFuns=[] # the functions, not the strings
        for i in range(self.nLayers):
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
            
            # nudge by random factor between -2 and 2
            wnudge=4*np.random.rand(self.layerSize[i+1],self.layerSize[i]) -2.
            bnudge=4*np.random.rand(self.layerSize[i+1]) - 2.
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
    
def fitness(snake,nx,ny,nutrition=100,gui=False,imgDir='gen'):
    # create an environment with one snoke
    # step in time until the snake dies, return score
    
    # snake is one instance of the class Snake
    # nx, ny are the number of fields in each direction
    # nutrition ... how many steps a snake gains by eating one piece of food
    
    if gui:
        mkdir(imgDir)
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
    n=2 # length of snake. i.e. snake_[:n] are non-negative locations
    
    sensor=np.zeros(10) 
        # 4 position of head-1
        # dist to 4 walls
        # 2 dist to food
    ll=0 # time step
    alive=True
    score=0.
    maxlifespan=2*nutrition
    while alive and ll<maxlifespan:
        
        if gui:
            saveField(score,nx,ny,sx,sy,foodx,foody,theFile=imgDir+'/%08i.png' %ll)
            
        # sense:
        sensor[0:4]=0. # wipe
        sensor[direction]=1. # true for direction of head-1
        sensor[4]=ny-1-sy[0] # number of free fields to the north
        sensor[5]=nx-1-sx[0] # number of free fields to the east
        sensor[6]=sy[0] # number of free fields to the south
        sensor[7]=sx[0] # number of free fields to the west
        sensor[8]=foodx-sx[0]
        sensor[9]=foody-sy[0]
        
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
        
        # check its not walked into a wall:
        if sx[0]==-1 or sy[0]==-1 or sx[0]==nx or sy[0]==ny:
            alive=False
            
        # periodic domain:
#        if sx[0]==-1: sx[0]=nx-1
#        if sy[0]==-1: sy[0]=ny-1
#        if sx[0]==nx: sx[0]=0
#        if sy[0]==ny: sy[0]=0
        
        # check its not walked into itself:
        #TODO check everywhere
        if sx[0]==sx[2] and sy[0]==sy[2]:
            alive=False
            
        if sx[0]==foodx and sy[0]==foody: # ate
            n+=1 # increment snake length
            score+=0.5
            maxlifespan+=nutrition
            # check if completed the game
            if n==(nx*ny):
                alive=False
            # create new food outside of snake 
            # TODO without guessing where the snake is and whole snake, not just head
            while sx[0]==foodx and sy[0]==foody:
                foodx=np.random.randint(low=0,high=nx)
                foody=np.random.randint(low=0,high=ny)
            
        else: #did not eat
            sx[n]=-9
            sy[n]=-9 # no longer part of the snake
        
        # fractional score as life time award for a total of 1 per food
        if alive:
            score += 0.5/nutrition
        
        ll +=1 # end of time step
    
    if gui:
        saveScoreImg(score,nx,ny,theFile=imgDir+'/%08i.png' %(ll))
    
    return score

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
    
    # network input and output layers
    layers=np.append(np.append(10,hiddenLayers),4)
    
    # build initial population
    pop=[]
    for i in range(popsize):
        pop.append(Snake(layers))
    
    popnew=np.copy(pop) # initialize
    for gen in range(ngen):
        
        # get scores
        scores=np.zeros(popsize)
        for i in range(popsize):
            scores[i]=fitness(pop[i],nx,ny,nutrition,gui=False,imgDir='')**2. # SQUARED
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
        print('Gen=%i, highscore=%f, median score=%f, mean score=%f' %(gen,scores[winner], np.median(scores), np.mean(scores)))
        if np.mod(gen,plotFreq)==0.: # let the best snake run again in a new scenario
            fitness(pop[winner],nx,ny,nutrition,gui=True,imgDir=baseDir+'/gen%06i' %gen)
            vFileName='gen%06i' %gen
            with open(baseDir+'/gen%06i/makeVideo.sh' %gen, 'w') as fid:
                fid.write('#!/bin/bash \nffmpeg -framerate 2 -i %08d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+vFileName+'.mp4 \n')
        
#        pop=popnew
        pop=np.copy(popnew)
    
    return

def trainReplace(nx,ny,popsize,ngen,mutationRate,hiddenLayers,activations,nutrition,baseDir,plotFreq=9e9):
    # replace low scoring individuals by mutations of high scoring ones
    
    # network input and output layers
    layers=np.append(np.append(10,hiddenLayers),4)
    
    # build initial population
    pop=[]
    for i in range(popsize):
        pop.append(Snake(layers))
    
    for gen in range(ngen):
        
        # get scores
        scores=np.zeros(popsize)
        for runs in range(10):
            for i in range(popsize):
                scores[i] += fitness(pop[i],nx,ny,nutrition,gui=False,imgDir='')
        ind=np.argsort(scores)[::-1] # indeces of individuals, starting with highest score
        
        # replace loser half of population
        for i in range(int(np.floor(popsize/2.))):
            iwinner=ind[i]
            iloser=ind[popsize-1-i]
            pop[iloser]=deepcopy(pop[iwinner])
            pop[iloser].mutate(rate=mutationRate)
            
        
        # plot highest score of each gen
        winner=ind[0]
        print('Gen=%i, highscore=%f, median score=%f, mean score=%f' %(gen,scores[winner], np.median(scores), np.mean(scores)))
        if np.mod(gen,plotFreq)==0.: # let the best snake run again in a new scenario
            fitness(pop[winner],nx,ny,nutrition,gui=True,imgDir=baseDir+'/gen%06i' %gen)
            vFileName='gen%06i' %gen
            with open(baseDir+'/gen%06i/makeVideo.sh' %gen, 'w') as fid:
                fid.write('#!/bin/bash \nffmpeg -framerate 2 -i %08d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+vFileName+'.mp4 \n')
        
    
    return




trainReplace(nx=20,ny=20,
      popsize=2000,
      ngen=101,
      mutationRate=0.1,
      hiddenLayers=np.array([30,30]),
#      activations=np.array(['relu','relu']),
#      activations=np.array(['sigmoid','sigmoid']),
      activations=np.array(['tanh','tanh']),
      nutrition=50, # steps gained per food. should correspond to map size
      baseDir='/home/sebastian/tmp/snake/run0',
      plotFreq=10)
