import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


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



class SVGD_model():

    def __init__(self):
        pass

    def SVGD_kernal(self, x, h=-1):
        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = h ** 2 / np.log(x.shape[0] + 1)

        kernal_xj_xi = np.exp(- pairwise_dists ** 2 / h)
        d_kernal_xi = np.zeros(x.shape)
        # for each particle i, this is computing sum_{j=1..N} nabla k(xi,xj)
        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = np.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

        return kernal_xj_xi, d_kernal_xi
    
    
    def matrices_for_Istein(self,x,lnprob,h=-1):
        
        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = h ** 2 / np.log(x.shape[0] + 1)
        # h=2sigma^2
            
        # KERNEL MATRIX : k(xi,xj) of size 100x100
        kernal_xj_xi = np.exp(- pairwise_dists ** 2 / h)
        

        # FIRST DERIVATIVES : nabla_xi k(xi,xj)= - (xi-xj) * k(xi,xj)
        d0=np.matrix(x - x[:,np.newaxis])
        d1=kernal_xj_xi* (-1 /h)
        d_kernal_xj_xi=np.multiply(d0,d1)
            
 
        # SECOND DERIVATIVES : nabla_{xi,xj} k(xi,x) = ( k(xi,x) + (xi-x)^2 * k(xi,x) ) * 2/h
        d1= kernal_xj_xi * 4 / h 
        d2= np.multiply(pairwise_dists**2, kernal_xj_xi) * 8/(h**2) 
        d2_kernal_xj_xi=(d1-d2) 
        
        # MATRIX OF SCORES of nabla log pi(xi)
        d_log_pi_xi=np.zeros(x.shape)
        for i_index in range(x.shape[0]):
            d_log_pi_xi[i_index]=lnprob(x[i_index])[0]
        
        return kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi
    

    def computeIstein(self,kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi):
        
        n = kernal_xj_xi.shape[0]
        d_log_pi_xj = d_log_pi_xi[:,np.newaxis]

        D = np.zeros((n,n))
        
        # first term with second derivative
        A = d2_kernal_xj_xi
        D+=A
        
        # cross terms
        b1= d_log_pi_xj - d_log_pi_xi
        b1=b1.reshape((n,n))
        B=np.multiply(b1,d_kernal_xj_xi)
        D+=B
        
        # gradloppi *k
        dot_product_d_log_pi=np.multiply(d_log_pi_xi, d_log_pi_xj)
        dot_product_d_log_pi=dot_product_d_log_pi.reshape((n,n))
        C=np.multiply(dot_product_d_log_pi,kernal_xj_xi)
        D+=C
        
        istein=np.sum(D)/(n**2)
        return istein
    


    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        x = np.copy(x0)
        Istein = [0] * n_iter
        KL = [0] * n_iter


        # adagrad with momentum
        eps_factor = 1e-8
        historical_grad_square = 0
        for iter in range(n_iter):
            if (iter)%10==0:
                print (iter)
            
            # compute Istein
            kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi = self.matrices_for_Istein(x,lnprob,h=-1)
            istein=self.computeIstein(kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi) #compute Istein
            Istein[iter]=istein
            
            #compute KL
            w = np.array([1/3, 2/3])
            mean = np.array([-2, 2])
            var = np.array([1, 1])

            OneDimensionGM_model = OneDimensionGM(w, mean, var)
            nb_bins=300
            x_lin = np.expand_dims(np.linspace(-15, 15, nb_bins), 1)

            mg_prob = OneDimensionGM_model.MGprob
            x_prob = mg_prob(x_lin)
            x_prob=x_prob/np.sum(x_prob)
            x_prob=x_prob[0:nb_bins-1]
            x_prob=x_prob.reshape((nb_bins-1,))
    
            y=np.histogram(x,bins=x_lin.reshape((nb_bins,)))[0]
            y=y / np.sum(y)
            #kl=KL2(y,x_prob)
            kl=entropy(y,x_prob)
            KL[iter]=kl
            
            # compute gradient and update
            kernal_xj_xi, d_kernal_xi = self.SVGD_kernal(x, h=-1)
            current_grad = (np.matmul(kernal_xj_xi, lnprob(x)) + d_kernal_xi) / x.shape[0]
            if iter == 0:
                historical_grad_square += current_grad ** 2
            else:
                historical_grad_square = alpha * historical_grad_square + (1 - alpha) * (current_grad ** 2)
            adj_grad = current_grad / np.sqrt(historical_grad_square + eps_factor)
            x += stepsize * adj_grad
            
        return x, Istein, KL


