"""
My attempt at deep Q learning
Using a NN to predict Q
The 'deep' part might not be justified because my NN doesnt have convolutions

Heavily inspired by deeplizard.com

Seb, Nov 2020
"""



import numpy as np

class mydqn():
    # class to train in a given environment
    
    def __init__(self,env,Q,verbose=False):
        # env is an instance of an environment class
        # Q is an instance of the NN that will approximate Q
        
        self.env=env
        self.Q=Q
        self.verbose=verbose
        self.memory={'state':[], 'rewards':[]}
        self.epsilon=1. # exploration rate initialised as 1.0
        self.gamma=0.8 # discount factor of future rewards
        
        return

    def episode(self):
        # play one episode, remember what happend to learn from later
        
        # start a game by getting the inital state of the environment
        state,gameover=self.env.initialise()
        if self.verbose:
            self.env.render()
        
        score=0. # accumulated rewards
        actions=[]
        rewards=[]
        while not gameover:
            # decide on exploitation or exploration and determine action
            if np.random.rand(1)>self.epsilon: # exploit, be greedy
                prediction=self.Q.forwardProp(state)
                action=np.argmax(prediction)
            else: # random step to explore
                action=np.random.randint(low=0,high=self.env.nActions) # draws in from range(0,nActions)
            
            # remember the state prior to taking the step
            self.memory['state'].append(state)
            actions.append(action)
            
            # take a step in the game
            state, gameover, reward=self.env.step(action)
            rewards.append(reward)
            score+=reward
            if self.verbose:
                self.env.render()
                print('Step reward = %f' %(reward))
        
        # reward to memorize is G, which is a function of the rewards of all (some) of the future rewards 
        # in this episode G is zero for all actions not taken, because we cant learn from those 
        # i.e. the gradient descent wont change the network when the loss is zero
        # it's not that those predictions were perfect, but we just cant learn from actiones never taken
        nsteps=len(actions)
        for ll in range(nsteps):
            G=np.zeros(self.env.nActions)
            for ill in range(ll,nsteps): # for all steps taken hereafter
                # the discount factor gamma decreases for steps taken further into the future
                # for this step, number ill, gamma is one. For future steps, it's gamma, gamma**2, gamma**3 and so on
                g=rewards[ill] * self.gamma**(ill-ll)
                G[actions[ll]]+=g
            self.memory['rewards'].append(G) # remember the discounted reward at this step
            if self.verbose:
                print('Discounted reward for step %i = %f' %(ll,G[actions[ll]]))
            
        
        return score # accumulated rewards
    
    def train(self,nEpisodes=1000,nbatch=None,nTraining=1):
        # run a few episodes to gather memoroes, then batch-train the NN
        # learning from random samples is more efficient than sequential (deeplizard)
        # nbatch ... how many batches in training of nEpisodes
        # nTraining ... how often to repeat the batch training process of the results of nEpisodes
        
        if nbatch==None:
            nbatch=int(nEpisodes/10.) 
        
        # gather data:
        scores=np.zeros(nEpisodes)
        for ee in range(nEpisodes):
            if self.verbose:
                print('Starting episode %i of %i' %(ee+1,nEpisodes))
            scores[ee]=self.episode()
        
        print('Finished gathering data from %i episodes. Mean score = %f. Max score = %f' %(nEpisodes,np.mean(scores),np.max(scores)))
        
        # learn from random data samples:
#        nTotal=len(self.memory['state'])
#        sel=np.arange(nTotal-nEpisodes,nTotal)
        for epoch in range(nTraining): # may repeat batch training
#            np.random.shuffle(sel) # shuffle range in place
#            self.Q.batchTrain1([self.memory['state'][i] for i in sel], [self.memory['rewards'][i] for i in sel])
#            self.Q.trainOneEpoch([self.memory['state'][i] for i in sel], [self.memory['rewards'][i] for i in sel], nbatch=nbatch)
            self.Q.trainOneEpoch(self.memory['state'][-nEpisodes:], self.memory['rewards'][-nEpisodes:], nbatch=nbatch) # will be shuffled there
        if self.verbose:
            self.Q.printWeightStats()
        
        # train a bit more on positive rewards
        if 0:
            sel=np.where(np.array(self.memory['rewards'])>0)[0]
            print('Training additionally on %i positive reward memories' %len(sel))
            self.Q.batchTrain1([self.memory['state'][i] for i in sel], [self.memory['rewards'][i] for i in sel])
        
        return







