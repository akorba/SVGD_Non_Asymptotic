import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from SVGD import SVGD_model
from scipy.spatial.distance import pdist, squareform


sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )

class OneDimensionGM():

    def __init__(self, omega, mean, var):
        self.omega = omega
        self.mean = mean
        self.var = var

    def dlnprob(self, x):
        rep_x = np.matlib.repmat(x, 1, self.omega.shape[0])
        category_prob = np.exp(- (rep_x - self.mean) ** 2 / (2 * self.var)) / (np.sqrt(2 * np.pi * self.var)) * self.omega
        den = np.sum(category_prob, 1)
        num = ((- (rep_x - self.mean) / self.var) * category_prob).sum(1)
        return np.expand_dims((num / den), 1)

    def MGprob(self, x):
        rep_x = np.matlib.repmat(x, 1, self.omega.shape[0])
        category_prob = np.exp(- (rep_x - self.mean) ** 2 / (2 * self.var)) / (np.sqrt(2 * np.pi * self.var)) * self.omega
        den = np.sum(category_prob, 1)
        return np.expand_dims(den, 1)

if __name__ == "__main__":

    
    def computeIstein(x,lnprob, h=-1):
        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = h ** 2 / np.log(x.shape[0] + 1)

        #kernel
        kernal_xj_xi = np.exp(- pairwise_dists ** 2 / h)
        
        #gradient kernel
        d_kernal_xi = np.zeros(x.shape)

        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = np.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h
            #d_kernal_xi_x_j[i_index] = np.matmul(x[i_index] - x, x[i_index] - x) * 2 / h

        #twice gradient
        d1= kernal_xj_xi** 2 / h
        d2=pairwise_dists ** 2 / h
        d_kernal_xi_x_j = d1+np.matmul(d1,d2)
        
        
        istein=0
        for i_index in range(x.shape[0]):
            for j_index in range(x.shape[0]):
                a=d_kernal_xi_x_j[i_index,j_index]
                istein+=a/ x.shape[0]**2
                b=-2*np.dot(lnprob(x[i_index]), d_kernal_xi[j_index])
                istein+=b/ x.shape[0]**2
                c=np.dot(lnprob(x[i_index]),lnprob(x[j_index]))[0,0]*kernal_xj_xi[i_index,j_index]
                istein+=c / x.shape[0]**2
                
        
        return istein
    
    
    
    
    
    
    w = np.array([1/3, 2/3])
    mean = np.array([-2, 2])
    var = np.array([1, 1])

    OneDimensionGM_model = OneDimensionGM(w, mean, var)

    np.random.seed(0)

    x0 = np.random.normal(-10, 1, [100, 1]);
    dlnprob = OneDimensionGM_model.dlnprob

    svgd_model = SVGD_model()
    
    
    n_iter = 500
    
    Istein= [0] * n_iter

    
    for i in range(n_iter):
        print (i)
        x = svgd_model.update(x0, dlnprob, n_iter=1, stepsize=1e-1, bandwidth=-1, alpha=0.9, debug=True)
        x0=x
        istein=computeIstein(x,dlnprob,h=-1)
        Istein[i]=istein
        

    #plot result
    sns.kdeplot(x.reshape((100,)), bw = .4, color = 'g')

    mg_prob = OneDimensionGM_model.MGprob
    x_lin = np.expand_dims(np.linspace(-15, 15, 100), 1)
    x_prob = mg_prob(x_lin)
    plt.plot(x_lin, x_prob, 'b--',color='red')
    plt.axis([-15, 15, 0, 0.4])
    plt.title(str(n_iter) + '$ ^{th}$ iteration')
    plt.show()

    plt.title('Stein Fisher divergence along iterations')
    plt.plot(Istein)
    plt.savefig('Istein500iterations.png')
    plt.show()
    np.save('data500.npy', Istein)
    #new_num_arr = np.load('data.npy') # load


