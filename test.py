#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from SVGD import SVGD_model, OneDimensionGM

sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )



if __name__ == "__main__":

    # target distribution
    w = np.array([1/3, 2/3])
    mean = np.array([-2, 2])
    var = np.array([1, 1])
    OneDimensionGM_model = OneDimensionGM(w, mean, var)
    dlnprob = OneDimensionGM_model.dlnprob


    np.random.seed(0)
    
    # run SVGD
    n_particles=200

    x0 = np.random.normal(-10, 1, [n_particles, 1]);
    svgd_model = SVGD_model()
    n_iter = 500
    x, Istein, KL = svgd_model.update(x0, dlnprob, n_iter=n_iter, stepsize=1e-1, bandwidth=-1, alpha=0.9, debug=True)

    
    #save vectors
    #np.save('saved_data/particles0.npy',x0)
    #np.save('saved_data/particles.npy',x)
    #np.save('saved_data/Istein.npy', Istein) 
    #np.save('saved_data/KL.npy', KL) 


    
    # 0) plot particles
    x_lin = np.expand_dims(np.linspace(-15, 15, n_particles), 1)
      
    mg_prob = OneDimensionGM_model.MGprob
    x_prob = mg_prob(x_lin)
    plt.plot(x_lin, x_prob, 'b--',color='b')
    
    
    sns.kdeplot(x0.reshape((n_particles,)), bw = .4, color = 'g')
    plt.axis([-15, 15, 0, 0.4])
    plt.title('Initial distribution')
    plt.grid()
    #plt.xlabel('Position of particles')
    plt.ylabel('Probability of the position')
    
    #plt.savefig('saved_data/particles.pdf')
    plt.show()
    
    plt.plot(x_lin, x_prob, 'b--',color='b')
    sns.kdeplot(x.reshape((n_particles,)), bw = .4, color = 'g')
    plt.axis([-15, 15, 0, 0.4])
    plt.title(str(n_iter) + '$ ^{th}$ iteration')
    plt.grid()
    plt.xlabel('Position of particles')
    plt.ylabel('Probability of the position')

    #plt.savefig('saved_data/particles500.pdf')
    plt.show()
    
    
    
    # 1) plot KL
    KL=KL[0:200]
    exprate=[(KL[0])*(0.99)**i for i in range(len(KL))]

    # in log scale
    plt.yscale('log')

    plt.plot(KL, label='KL')
    plt.plot(exprate, label='0.99^{n}')

    plt.legend(loc='best')
    plt.title('Evolution of KL')
    plt.ylabel('KL between current iterate and pi')
    plt.xlabel('Number of iterations')
    plt.grid()

    #plt.savefig('saved_data/KLlog.pdf')
    plt.show()
 
    
    # 2) plot Istein     
    Isteinavg=[np.mean(Istein[0:i]) for i in range(1,len(Istein)+1)]
    theoreticalrate=[(Istein[0])/(i+1) for i in range(len(Istein))]
    
    # in log scale
    plt.xscale('log')
    plt.yscale('log')
    
    plt.plot(Isteinavg, label='Avg-Istein')
    plt.plot(theoreticalrate,label='1/n')
    
    plt.legend(loc='best')
    plt.title('Evolution of Istein')
    plt.ylabel('IStein between current iterate and pi')
    plt.xlabel('Number of iterations')
    plt.grid()

    #plt.savefig('saved_data/Isteinlog.pdf')
    plt.show()
        
    
        
            
    
    
    
