# testing a neural net on predicting the return of exclusive or
import numpy as np
from annvanilla import ann


# define data
x=dict.fromkeys(range(4))
y=dict.fromkeys(range(4))
x[0]=np.array([0,1]); y[0]=np.array([1])
x[1]=np.array([1,0]); y[1]=np.array([1])
x[2]=np.array([0,0]); y[2]=np.array([0])
x[3]=np.array([1,1]); y[3]=np.array([0])

# init
model=ann(2,1,np.array([2]),alpha=0.1)
#    print model.layerSize

# train
nt=50000
logy=np.zeros((nt,4)) # logging output as I train
for l in range(nt):
    i=np.random.randint(low=0,high=4) # int in [0,4)
    
    cost=y[i]-model.forwardProp(x[i])
    model.backwardProp(cost)
    
    # test:
    for i in range(4):
        logy[l,i]=model.forwardProp(x[i])
    
    
# test
for i in range(4):
    print(x[i], y[i], model.forwardProp(x[i]))
#    print(model.b)
    
import matplotlib.pyplot as plt
plt.plot(np.arange(nt)+1,logy[:,0],'k-',label='x=[0,1]')
plt.plot(np.arange(nt)+1,logy[:,1],'r-',label='x=[1,0]')
plt.plot(np.arange(nt)+1,logy[:,2],'g-',label='x=[0,0]')
plt.plot(np.arange(nt)+1,logy[:,3],'b-',label='x=[1,1]')
plt.legend()
plt.xlabel('learning iteration')
plt.ylabel('y'); plt.ylim([0,1])
plt.title('alpha=%f' %model.alpha)
if 0:
    plt.savefig('xorTest_alpha=%.4f.png' %model.alpha,dpi=400,bbox_inches='tight')
    