#!/usr/bin/env python3

import numpy as np
from mydqn import mydqn
import sys
sys.path.append('../annvanilla')
from annvanilla import ann

# set up the game environment
from environment0 import environment
env=environment(randomise=True)


# set up the NN
nin=env.n
nout=env.nActions
hiddenLayers=np.array([20])
activFuns=np.array(['relu'])
alpha=0.1 # learning rate
lambd=0.001 # regularization
Q=ann(nin,nout,hiddenLayers,alpha=alpha,lambd=lambd,activFuns=activFuns)

# set up Q learning in the environment
game=mydqn(env,Q,verbose=0)


# train the agent
nEpisodes=20 # in each set
nSets=30
with open('training.log','w') as fid:
    fid.write('# Set, avg score across %i episodes\n' %nEpisodes)
for i in range(nSets):
    game.train(nEpisodes)
    game.epsilon*=0.95 # reduce exploration rate
    
    # run a test to see how good the agent is without exploration
    eps=game.epsilon # save this to reset after
    scores=np.zeros(nEpisodes)
    game.epsilon=0. # no exploration
    for ee in range(nEpisodes):
        scores[ee]=game.episode()
#    print('Set=%i, scores exploiting = ' %i,scores)
    with open('training.log','a') as fid:
        fid.write('%i\t%f\n' %(i,np.mean(scores)))
    game.epsilon=eps # reset for training
    
    