#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from SVGD import OneDimensionGM


from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42



w = np.array([1/3, 2/3])
mean = np.array([-2, 2])
var = np.array([1, 1])

OneDimensionGM_model = OneDimensionGM(w, mean, var)
np.random.seed(0)

x0 = np.random.normal(-10, 1, [100, 1]);
y0 = np.random.normal(-10, 1, [100, 1]);
dlnprob = OneDimensionGM_model.dlnprob

fontP = FontProperties()
fontP.set_size('medium')
legend = plt.legend(loc = 1, shadow = True, fontsize = 'x-large', prop = fontP)

mg_prob = OneDimensionGM_model.MGprob
x_lin = np.expand_dims(np.linspace(-15, 15, 100), 1)
x_prob = mg_prob(x_lin)

plt.plot(x_lin, x_prob, 'b--',color='b', label='target')

# plot initial particles
sns.kdeplot(x0.reshape((100,)), bw = .4, color = 'g', label='svgd particles')
plt.axis([-15, 15, 0, 0.4])
plt.title('Initial distribution of particles')
plt.show()
sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )


# load data
x=np.load('saved_data/particles.npy')
Istein=np.load('saved_data/Istein.npy')
KL=np.load('saved_data/KL.npy')


n_iter=len(Istein)


# plot particles
fontP = FontProperties()
fontP.set_size('medium')
legend = plt.legend(loc = 1, shadow = True, fontsize = 'x-large', prop = fontP)

mg_prob = OneDimensionGM_model.MGprob
x_lin = np.expand_dims(np.linspace(-15, 15, 100), 1)
x_prob = mg_prob(x_lin)
plt.plot(x_lin, x_prob, 'b--',color='b', label='target')
# plot final particles
sns.kdeplot(x.reshape((100,)), bw = .4, color = 'g', label='svgd particles')

plt.axis([-15, 15, 0, 0.4])
plt.title(str(n_iter) + '$ ^{th}$ iteration')
#plt.savefig('saved_data/particles.pdf')
plt.show()

# plot Istein


Isteinavg=[np.mean(Istein[0:i]) for i in range(1,len(Istein)+1)]
#Isteinmin=[np.min(Istein[0:i]) for i in range(1,len(Istein)+1)]
theoreticalrate=[(Istein[0])/(i+1) for i in range(len(Istein))]
smallerrate=[(Istein[0])/(np.sqrt(i+1)) for i in range(len(Istein))]

# in log scale
plt.xscale('log')
plt.yscale('log')

plt.plot(Istein, label='Istein')
plt.plot(Isteinavg, label='Avg-Istein')
#plt.plot(Isteinmin, label='Min-Istein') #no oscillations so Istein and Isteinmin are the same
plt.plot(theoreticalrate,label='1/n')
plt.plot(smallerrate,label='1/sqrt(n)')

plt.legend(loc='best')
plt.title('Evolution of Istein (log scale)')
#plt.savefig('saved_data/Isteinlog.pdf')
plt.show()


# without log scale

Istein=Istein[0:500]
Isteinavg=[np.mean(Istein[0:i]) for i in range(1,len(Istein)+1)]
#Isteinmin=[np.min(Istein[0:i]) for i in range(1,len(Istein)+1)]
theoreticalrate=[(Istein[0])/(i+1) for i in range(len(Istein))]
smallerrate=[(Istein[0])/(np.sqrt(i+1)) for i in range(len(Istein))]

plt.plot(Istein, label='Istein')
plt.plot(Isteinavg, label='Avg-Istein')
#plt.plot(Isteinmin, label='Min-Istein') #no oscillations so Istein and Isteinmin are the same
plt.plot(theoreticalrate,label='1/n')
plt.plot(smallerrate,label='1/sqrt(n)')

plt.legend(loc='best')
plt.title('Evolution of Istein')
#plt.savefig('saved_data/Istein.pdf')

plt.show()


# plot KL
    


"""



%matplotlib inline
papc = plt.plot(PAPC, label = 'PAPC', linestyle = '--')
extra = plt.plot(EXTRA2, label = 'Revisiting EXTRA', linestyle = '--')
papcacc = plt.plot(PAPCacc, label = 'Algorithm 1', linewidth = 4.0)
extraacc = plt.plot(EXTRAacc, label = 'Accelerated EXTRA', linestyle = '--')
scary = plt.plot(SCARY, label = 'Algorithm 2', linewidth = 4.0)
papcopt = plt.plot(PAPCopt, label = 'Algorithm 3', linewidth = 4.0)
ssda = plt.plot(SSDA, label = 'SSDA', linestyle = '--')
msda = plt.plot(MSDA, label = 'MSDA', linestyle = '--')
penaltyacc = plt.plot(PENALTYacc, label = 'Accelerated Penalty', linestyle = '--')



plt.yscale('log')
fontP = FontProperties()
fontP.set_size('medium')
legend = plt.legend(loc = 1, shadow = True, fontsize = 'x-large', prop = fontP)
plt.title('Gradient calls')
plt.ylabel('distance between current iterate and $x^*$')
plt.xlabel('number of gradient calls')
plt.grid()
plt.savefig('gradient_calls_N_'+str(N)+'_M_'+str(M)+'_std_'+str(std)+'.png')

plt.show()

from numpy import linalg as LA

"""





