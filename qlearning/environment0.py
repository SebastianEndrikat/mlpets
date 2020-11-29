

# define the learning environemt (the game board)

import numpy as np

class environment():
    """
    This is the simplest little game possible
    A tiny 1D domain. Agent starts in the middle
    Reaching a side kills. 
    In one direction, a high reward ends the game
    In the other direction a low reward ends the game
    Taking a step that doesnt end the game has a slightly negative reward.
    Which side is high or low could be randomized to 
    make training more difficult (randomise=True).
    """
    
    def __init__(self,randomise=False):
        self.randomise=randomise
        
        # some info about this game:
        self.nmaxsteps=100
        self.nx=11 # board size in x int
        self.n=self.nx # total number of state elements. In this case, same as nx
        self.nActions=2
        
        
        return
    
    def step(self,action):
        # action is the argmax of the prediction. 
        # Define what that means in the environment here
        
        gameover=False
        
        # take the step
        if action==0:
            self.ii-=1 # move left
        elif action==1:
            self.ii+=1 # move right
        else:
            raise ValueError('Undefined action was passed to environment.step()')
            
        # see if that landed the agent on a ghost (out of bounds) cell
        if self.ii==0 or self.ii==(self.nx-1):
            gameover=True
        
        # check if time ran out
        self.ll+=1
        if self.ll>self.nmaxsteps:
            gameover=True
        
        reward=self.rewards[self.ii]
        self.state=np.copy(self.board) # get a clean board
        self.state[self.ii]=self.valAgent # mark the new agent position
        
        return self.state, gameover, reward
    
    def initialise(self):
        # return the initial state of a new game
        # this is different from __init__, because it starts a new 
        # game, rather than a new environment
        
        
        # define state values (encoding the domain layout):
        self.valAgent=1.
        self.valGhost=-1. # ghost-cell on the side, i.e. field that kills
        self.valPrize=10.
            
        
        # define the state, i.e. the layout of the game board
        # high and low reward regions have to be marked 
        # set up the rewards gained on each field in the domain
        if not self.randomise:
            self.setlayoutA()
        else:
            if np.random.rand(1)>=0.5:
                self.setlayoutA()
            else:
                self.setlayoutB()
        
        self.ll=0 # step count in the game
        self.ii=int(np.floor(self.nx/2.)) # initial position in the domain
        gameover=False
        
        # create the state by copying the board and marking the agent 
        # in its current position
        self.state=np.copy(self.board)
        self.state[self.ii]=self.valAgent
        
        
        return self.state,gameover
    
    def setlayoutA(self):
        self.board=np.zeros(self.nx)
        self.rewards=-np.ones(self.nx)
        self.rewards[0]=-10. # low reward side
        self.board[0]=self.valGhost
        self.rewards[self.nx-1]=10. # high reward side
        self.board[self.nx-1]=self.valPrize
        return
    def setlayoutB(self):
        self.board=np.zeros(self.nx)
        self.rewards=-np.ones(self.nx)
        self.rewards[self.nx-1]=-10. # low reward side
        self.board[self.nx-1]=self.valGhost
        self.rewards[0]=10. # high reward side
        self.board[0]=self.valPrize
        return
    
    def render(self):
        # save an image of the current state
        # TODO implement image redering
        print(self.state)
        
        return
    
    
    
    