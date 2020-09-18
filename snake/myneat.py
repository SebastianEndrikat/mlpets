#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 07:50:35 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
    
    
An algorithm that evolves the network strucutre and its weights
based on a provided fitness function
    
"""


import numpy as np


class neat():
    
    def __init__(self,nin,nout,popsize,evalFitness):
        # evalFitness is a function that takes an instance of
        # this class and returns an array of scores, popsize long
        
        self.nin=nin
        self.nout=nout
        self.popsize=popsize
        self.evalFitness=evalFitness
        
        # nodes
        self.nodes=np.zeros(nin+nout)
        self.iin=range(nin) # indeces of input nodes
        self.nnodes=nin+nout
        self.iout=range(nin,self.nnodes) # indeces of output nodes
        
        # connections
        # inital connection only from the first input to the first output
        self.c0=np.array([0]) # connections from
        self.c1=np.array([nin]) # connections to
        self.nconn=1 # total number of unique connections
        
        # one weight per connection for every member of the population
        # weight of 0 means connection is inactive
#        self.w=np.random.rand(self.nconn,self.popsize)
        self.w=np.zeros((self.nconn,self.popsize))
        for i in range(self.popsize):
            self.w[:,i]=self.getnNewWeightValues(self.nconn)
        
        # activations
        # every node gets an activation for every member of the population
        # is only used for non-input nodes tho in predict
        # note that depending on the activation, nodes that have no input from anywhere
        # can still have a non-zero value that does not depend on the network input
        self.defaultActivs=[]
        for i in range(self.popsize):
            self.defaultActivs.append(self.relu) # one for each member of the population
        self.activations=[]
        for i in range(self.nnodes):
            self.activations.append(self.defaultActivs) # call it as activations[nodeIndex][populationMember]
        
        return
    
    
    def sigmoid(self,x):
        return 1./(1.+np.exp(-x)) # logistic function
    def relu(self,x):
        # only single floats are passed to this version
        return np.max([0.,x])
    def tanh(self,x):
        return np.tanh(x)
    
    def getnNewWeightValues(self,n):
        
        eps=(3.0/self.nconn)**0.5
        w=(2.0*eps*np.random.rand(n)) - eps
        return w
    
    def addNode(self):
        
        self.nodes=np.append(self.nodes,0.)
        self.nnodes +=1
        self.activations.append(self.defaultActivs)
        
        return self.nnodes-1 # index of new node
    
    def addConnection(self,n0,n1):
        # connect nodes with indeces n0 and n1
        # unless that connection already exists
        
        if not self.nodesAreConnected(n0,n1):
            self.c0=np.append(self.c0,n0)
            self.c1=np.append(self.c1,n1)
            # weight of new connection is 0 for all members of the population:
            self.w=np.vstack((self.w,np.zeros((1,self.popsize))))
            j = self.nconn
            self.nconn+=1
        else:
            j=self.findConnection(n0,n1)
        
        return j # index of (new) connection
    
    def nodesAreConnected(self,n0,n1):
        # check if nodes n0 and n1 are connected
        # TODO speed up using numpy
        
        for i in range(self.nconn):
            if self.c0[i]==n0 and self.c1[i]==n1:
                return True
        return False
    
    def findConnection(self,n0,n1):
        # find the index of the connection between nodes n0 and n1
        # TODO speed up using numpy
        
        for i in range(self.nconn):
            if self.c0[i]==n0 and self.c1[i]==n1:
                return i
        return -1
    
    def predict(self,imem,vals,TOL=1e-4,llmax=2e3):
        # imem is the index of the member who predicts
        # vals are teh values of the input nodes
        self.nodes*=0. # reset
        self.nodes[self.iin]=vals
        # TODO pull some things out to not repeat all the time
        
        outold=np.sum(self.nodes[self.iout])
        res=9e9
        ll=0
        while res>TOL and ll<llmax:
            # for every non-input node, find all connections that end there and evaluate its value
            for i in range(self.nin,self.nnodes):
                # get indeces of connections that end in this node
                sel,=np.where(self.c1==i)
                self.nodes[i]=0.
                for j in sel: # indeces of connections that end at node i
                    self.nodes[i]+=(self.w[j,imem]*self.nodes[self.c0[j]]) 
                self.nodes[i]=self.activations[i][imem](self.nodes[i])
            outnew=np.sum(self.nodes[self.iout])
            res=np.abs(outnew-outold)
            outold=outnew
            ll+=1
                
        return self.nodes[self.iout]
    
    def mutateWeights(self,imem,rate):
        # imem is the index of the member whos weights are randomly mutated
        # only mutating active weights, i.e. not enabling connections with w=0
        activeConns=(self.w[:,imem] != 0.)
        nactive=np.sum(activeConns)
        mut=(np.random.rand(nactive)<=rate) # which ones are on for mutation
        
        # mutate by factor:
#        factors=3*np.random.rand(nactive) -1.5 # mutation means *=[-1.5,1.5]
#        self.w[activeConns,imem] *= (factors*mut)
        
        # mutate by addition:
        therange=np.std(self.w[activeConns,imem])*2.
        additions=therange*np.random.rand(nactive) - (therange/2.)
        self.w[activeConns,imem] += (additions*mut)
        
        return
    
    def mutateConnection(self,imem):
        # mutate one connection for one member
        # 1. decide if add or delete connection
        # 2.1 if delete, just set that weight to zero
        # 2.2 if add, check if connection exists, else create it. Assign random weight
        
        nactiveConns=np.sum(self.w[:,imem] != 0.)
        # fraction of active connections sets likelihood of adding vs deleting a connection
        addC=(np.random.rand(1)>(nactiveConns/float(self.nconn)))
        
#        addC=(np.random.rand(1)>0.48) # sets likelihood of adding vs deleting a connection
        if addC:
            # no need to check if the network is fully connected and no unique new
            # connection can be added, because
            # we might well 'create' a connection that already exists. We simply
            # reset the weight for this member, potentially activating the connection
            n0=np.random.randint(self.nnodes)
            n1=np.random.randint(self.nnodes)
#            while self.nodesAreConnected(n0,n1): # repeat if necessary
#                n0=np.random.randint(self.nnodes)
#                n1=np.random.randint(self.nnodes)
            j=self.addConnection(n0,n1) # returns index of connection
#            self.w[j,imem]=np.random.rand(1) # TODO define random range in __init__ for all new weights
            self.w[j,imem]=self.getnNewWeightValues(1)
        else: # delete
            j=np.random.randint(self.nconn)
            self.w[j,imem]=0.
        
        return
    
    def replaceMember(self,imemDelete,imemCopy):
        # overwrite member number imemDelete with a copy of imemCopy
        
        self.w[:,imemDelete]=self.w[:,imemCopy]
        for i in range(self.nnodes):
            self.activations[i][imemDelete]=self.activations[i][imemCopy]
        
        return
    
    def resetMember(self,imemDelete):
        # overwrite member number imemDelete with random/default values
        
        self.w[:,imemDelete]=np.random.rand(self.nconn) # TODO some should be zero=deactivated
        for i in range(self.nnodes):
            self.activations[i][imemDelete]=self.defaultActivs[0]
            
        return
    
    def train00(self,nepoch):
        
        stats_hs=np.zeros(nepoch)
        for e in range(nepoch):
            scores=self.evalFitness(self)
            stats_hs[e]=np.max(scores)
            print('%i: Highscore=%f, mean score = %f, median score = %f, nnodes=%i, nconn=%i' 
                  %(e,np.max(scores),np.mean(scores),np.median(scores),self.nnodes,self.nconn ))
            
            # debug:
#            nactiveConns=np.sum(self.w != 0., axis=0)
#            print('smallest network has %i connections' %np.min(nactiveConns))
            
            iranked=np.argsort(scores) # lowest first
            self.best=iranked[::-1]
            nRep=int(np.floor(self.popsize/2.)) # replace this many
            for i in range(nRep):
                self.replaceMember(iranked[i],self.best[i]) # replace low scoring with high scoring
                self.mutateWeights(iranked[i],rate=0.2)
                self.mutateConnection(iranked[i]) # likelihood of adding or deleting is set there 
            # mutate connection for the one that scored lowest but was replaced with 
            # mutated version of the highest scoring member:
            # TODO incentivise small networks
#            self.mutateConnection(iranked[0]) 
            
            # add new nodes 
            # TODO under certain cirumstances
            if (not np.mod(e,10) and e<=50):
                self.addNode()
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(nepoch)+1,stats_hs,'k.')
        
        return
    
#TODO
# mutate activations
# cross-over

# =============================================================================
# =============================================================================
if __name__ == "__main__":
    print('Executing this as a script...')

