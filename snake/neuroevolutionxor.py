#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 07:50:35 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
"""


import numpy as np
np.random.seed(42)
from myneat import neat


    
def fitness(neuro):
    # fitness function for this particular evolution
    # accepts an instance of neat, which contains predict
    
    x=( (0,1), (1,0), (0,0), (1,1) )
    y=(1,1,0,0)
    n=4
    
    scores=np.zeros(neuro.popsize)
    for imem in range(neuro.popsize):
        for i in range(n):
            # squared error cuz small error should not be a problem
            # as the result would be rounded in testing
            yp=neuro.predict(imem,x[i])
            err=(y[i]-yp)**2.
            scores[imem]+=( 1.-err )
            
    nactiveConns=np.sum(neuro.w != 0., axis=0) # number of used connections for every member of the population
    penalties=nactiveConns/float(neuro.nconn) # ratio of used to total number of connections
    
    return scores - penalties

def printFitness(neuro,ii=0):
    # for the iith best member
    
    x=( (0,1), (1,0), (0,0), (1,1) )
    y=(1,1,0,0)
    n=4
    imem=neuro.best[ii]
    nactiveConns=np.sum(neuro.w[:,imem] != 0.)
    print('Member number %i is using %i of %i connections between %i nodes.'
          %(ii,nactiveConns,neuro.nconn,neuro.nnodes))
    for i in range(n):
        pred=neuro.predict(imem,x[i])
        print('input: ', x[i], ' target=%i, prediction = %i (%.6f)' %(y[i], np.round(pred), pred))
    return
    
    
neuro=neat(nin=2,nout=1,popsize=500,evalFitness=fitness)



#print(neuro.predict(0,(0.5,0.5)))
#print(fitness(neuro))
neuro.train00(100)
scores=fitness(neuro)
printFitness(neuro)
