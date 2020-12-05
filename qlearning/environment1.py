

# define the learning environemt (the game board)

import numpy as np

class environment():
    """
    This is a simple 2D game environment.
    Agent starts in the middle
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
        self.nx=21 # board size in x int
        self.ny=11 # board size in y int
        self.n=self.nx*self.ny # total number of state elements
        self.nActions=4
        
        
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
        elif action==2:
            self.jj-=1 # move down
        elif action==3:
            self.jj+=1 # move up
        else:
            raise ValueError('Undefined action was passed to environment.step()')
            
        # see if that landed the agent on a ghost (out of bounds) cell
        if self.ii==0 or self.ii==(self.nx-1) or self.jj==0 or self.jj==(self.ny-1):
            gameover=True
        
        # check if time ran out
        self.ll+=1
        if self.ll>self.nmaxsteps:
            gameover=True
        
        reward=self.rewards[self.ii,self.jj]
        self.state=np.copy(self.board) # get a clean board
        self.state[self.ii,self.jj]=self.valAgent # mark the new agent position
        
        return self.state.ravel(), gameover, reward
    
    def initialise(self):
        # return the initial state of a new game
        # this is different from __init__, because it starts a new 
        # game, rather than a new environment
        
        
        # define state values (encoding the domain layout):
        self.valAgent=1.
        self.valGhost=-1. # ghost-cell on the side, i.e. field that kills
        self.valPrize=10.
        
        # define rewards to get
        self.rewardKill=-10. # out of bounds or somewhere in the domain 'low reward field'
        self.rewardPrize=(self.n*1.0)**0.5 # connected to the board size, because it takes more steps to get here on larger boards
            
        
        # define the state, i.e. the layout of the game board
        # high and low reward regions have to be marked 
        # set up the rewards gained on each field in the domain
        self.board=np.zeros((self.nx,self.ny))
        self.rewards=-np.ones((self.nx,self.ny)) # unless changed below, every step has -1 reward
        self.board[:, 0]=self.valGhost # mark ghost cells that are out-of-bounds
        self.board[:,-1]=self.valGhost
        self.board[0 ,:]=self.valGhost
        self.board[-1,:]=self.valGhost
        self.rewards[:, 0]=self.rewardKill # mark ghost cells that are out-of-bounds
        self.rewards[:,-1]=self.rewardKill
        self.rewards[0 ,:]=self.rewardKill
        self.rewards[-1,:]=self.rewardKill
        if not self.randomise:
            self.setlayoutA()
        else:
            if np.random.rand(1)>=0.5:
                self.setlayoutA()
            else:
                self.setlayoutB()
        
        self.ll=0 # step count in the game
        self.ii=int(np.floor(self.nx/2.)) # initial position in the domain
        self.jj=int(np.floor(self.ny/2.)) # initial position in the domain
        gameover=False
        
        # create the state by copying the board and marking the agent 
        # in its current position
        self.state=np.copy(self.board)
        self.state[self.ii,self.jj]=self.valAgent
        
        
        return self.state.ravel(), gameover
    
    def setlayoutA(self):
        self.rewards[1,1]=self.rewardKill # low reward side
        self.board[1,1]=self.valGhost # mark the low reward as a death hole, same as boudary
        self.rewards[self.nx-2,self.ny-2]=self.rewardPrize # high reward side
        self.board[self.nx-2,self.ny-2]=self.valPrize # mark high reward on the board
        return
    def setlayoutB(self):
        self.rewards[self.nx-2,self.ny-2]=self.rewardKill # low reward side
        self.board[self.nx-2,self.ny-2]=self.valGhost # mark the low reward as a death hole, same as boudary
        self.rewards[1,1]=self.rewardPrize # high reward side
        self.board[1,1]=self.valPrize # mark high reward on the board
        return
    
    def render(self):
        # save an image of the current state
        # TODO implement image redering
        for jj in range(self.ny-1,-1,-1):
            for ii in range(self.nx):
                if self.state[ii,jj]==self.valGhost: # killing field
                    print('X',end='')
                elif self.state[ii,jj]==self.valAgent:
                    print('O',end='')
                elif self.state[ii,jj]==self.valPrize:
                    print('@',end='')
                else:
                    print(' ',end='')
            print('') # new line
                
#            print(self.state[:,jj])
        
        return
    
    
    
    