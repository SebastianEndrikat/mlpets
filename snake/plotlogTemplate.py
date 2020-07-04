#!/usr/bin/env python3



import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl


wf=np.loadtxt('run.log')


mpl.rcParams['figure.subplot.wspace'] = 0.3 # width space
mpl.rcParams['figure.subplot.hspace'] = 0.25 # height space
mpl.rcParams['figure.subplot.left'] = 0.0
mpl.rcParams['figure.subplot.right'] = 1.
mpl.rcParams['figure.subplot.bottom'] = 0.0
mpl.rcParams['figure.subplot.top'] = 1


#mpl.rcParams['axes.formatter.limits'] = (-3,3)

textwidth=4.
figWidth=2.0*textwidth
figHeight = 0.8*figWidth
f,ax=plt.subplots(2,2,figsize=(figWidth,figHeight))#,sharey=True)

thisax=ax[0,0]
thisax.plot(wf[:,0],wf[:,1],'k.',label='high score')
thisax.plot(wf[:,0],wf[:,2],'b*',label='median score',markersize=3.)
thisax.plot(wf[:,0],wf[:,3],'rv',label='mean score',markersize=2.)
thisax.set_xlabel('generation')
thisax.set_ylabel('score')
thisax.set_ylim([0,25])
thisax.set_xlim([0,500])
thisax.legend()



thisax=ax[0,1]
thisax.plot((wf[:,4]-wf[0,4])/60.,wf[:,1],'k.',label='high score')
thisax.plot((wf[:,4]-wf[0,4])/60.,wf[:,2],'b*',label='median score',markersize=3.)
thisax.plot((wf[:,4]-wf[0,4])/60.,wf[:,3],'rv',label='mean score',markersize=2.)
thisax.set_xlabel('time in minutes on 1 core')
thisax.set_ylabel('score')
thisax.set_ylim([0,25])
thisax.set_xlim([0,200])
#thisax.legend()


thisax=ax[1,0]
thisax.plot(wf[:,0],wf[:,5],'k.',label='wall')
thisax.plot(wf[:,0],wf[:,6],'b*',label='canibalism',markersize=3.)
thisax.plot(wf[:,0],wf[:,7],'rv',label='starvation',markersize=2.)
thisax.set_xlabel('generation')
thisax.set_ylabel('fractional cause of death')
thisax.set_ylim([0,1])
thisax.set_xlim([0,500])
thisax.legend()


thisax=ax[1,1]
thisax.plot((wf[:,4]-wf[0,4])/60.,wf[:,5],'k.',label='wall')
thisax.plot((wf[:,4]-wf[0,4])/60.,wf[:,6],'b*',label='canibalism',markersize=3.)
thisax.plot((wf[:,4]-wf[0,4])/60.,wf[:,7],'rv',label='starvation',markersize=2.)
thisax.set_xlabel('time in minutes on 1 core')
thisax.set_ylabel('fractional cause of death')
thisax.set_ylim([0,1])
thisax.set_xlim([0,200])
#thisax.legend()


plt.savefig('stats.png',dpi=500,bbox_inches='tight')
