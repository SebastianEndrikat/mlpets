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
        
        return

    def episode(self):
        # play one episode, remember what happend to learn from later
        # TODO delayed reward: distribute rewards in memory after the episode. Thats where Bellman comes in?
        
        # start a game by getting the inital state of the environment
        state,gameover=self.env.initialise()
        if self.verbose:
            self.env.render()
        
        score=0. # accumulated rewards
        while not gameover:
            # decide on exploitation or exploration and determine action
            if np.random.rand(1)>self.epsilon: # exploit, be greedy
                prediction=self.Q.forwardProp(state)
                action=np.argmax(prediction)
            else: # random step to explore
                action=np.random.randint(low=0,high=self.env.nActions) # draws in from range(0,nActions)
            
            # remember the state prior to taking the step
            self.memory['state'].append(state)
            
            # take a step in the game
            state, gameover, reward=self.env.step(action)
            score+=reward
            if self.verbose:
                self.env.render()
                print('Step reward = %f' %(reward))
                
            # memorise what just happened:
            # in a form that the NN expects for backprop
            # TODO check if this makes sense: the target vector (i.e. the true reward),
            # has the received reward for the action taken and reward=0 for the action(s) 
            # that were not actually taken. Maybe they would have brought in a higher
            # reward but we dont know so its 0 ... does that make sense?
            rewards=np.zeros(self.env.nActions)
            rewards[action]=reward
            self.memory['rewards'].append(rewards)
        
        
        return score # accumulated rewards
    
    def train(self,nEpisodes=1000):
        # run a few episodes to gather memoroes, then batch-train the NN
        # learning from random samples is more efficient than sequential (deeplizard)
        
        # gather data:
        scores=np.zeros(nEpisodes)
        for ee in range(nEpisodes):
            if self.verbose:
                print('Starting episode %i of %i' %(ee+1,nEpisodes))
            scores[ee]=self.episode()
        
        print('Finished gathering data from %i episodes. Mean score = %f' %(nEpisodes,np.mean(scores)))
        
        # learn from data:
#        self.Q.printWeightStats()
        # TODO randomise samples 
        self.Q.batchTrain1(self.memory['state'][-nEpisodes:], self.memory['rewards'][-nEpisodes:])
        self.Q.printWeightStats()
        
        return







