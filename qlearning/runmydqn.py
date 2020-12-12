#!/usr/bin/env python3

import numpy as np
from mydqn import mydqn
import sys
sys.path.append('../annvanilla')
from annvanilla import ann

# set up the game environment
#from environment0 import environment
from environment1 import environment
#env=environment(randomise=True)
env=environment(randomise=False)


# set up the NN
nin=env.n
nout=env.nActions
hiddenLayers=np.array([20,20])
activFuns=np.array(['sigmoid','sigmoid'])
alpha=0.1 # learning rate
lambd=0.001 # regularization
Q=ann(nin,nout,hiddenLayers,alpha=alpha,lambd=lambd,activFuns=activFuns)

# set up Q learning in the environment
game=mydqn(env,Q,verbose=0)


# train the agent
nEpisodes=400 # in each set
nbatch=40
nSets=1000
with open('training.log','w') as fid:
    fid.write('# Set, avg score across %i episodes\n' %nEpisodes)
for i in range(nSets):
    game.train(nEpisodes,nbatch,nTraining=1)
    game.epsilon*=0.99 # reduce exploration rate
    
    # run a test to see how good the agent is without exploration
    eps=game.epsilon # save this to reset after
    scores=np.zeros(nEpisodes)
    COD=np.zeros(nEpisodes)
    game.epsilon=0. # no exploration
    for ee in range(nEpisodes):
        scores[ee]=game.episode()
        COD[ee]=env.causeOfDeath
#    print('Set=%i, scores exploiting = ' %i,scores)
    with open('training.log','a') as fid:
        fid.write('%i\t%f\t%i\t%i\t%i\n' %(i,np.mean(scores), np.sum(COD==1), np.sum(COD==2), np.sum(COD==3)))
    game.epsilon=eps # reset for training
    
    
##### NOTES
# ran environment0 with alpha=0.1 and lambda=0.001. 1 hidden layer with 20 and relu, 20 episodes and a few sets... done
    
# in environment1, weights blow up with relu in hidden layer. 
# regularization doesnt help. 
# sigmoid in all layers keeps weights under control
# training is not fruitful tho. Maybe the agent doesnt stumple upon the prize often enough
# training more on positive reward memories doesnt seem to help much
# switched to two hidden layers. doesnt help. agent is only learning to run straight into a wall
# batch training used too many samples at once. better now but still doesnt learn properly
# input state values too weird? normalize board description somehow?
    
    
