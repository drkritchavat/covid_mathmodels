import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from numpy.random import poisson
class LogNormal:
    def __init__(self,mu,sd):
        self.mu = mu
        self.sd = sd
        
    def random(self):
        return int(np.round(np.exp(np.random.normal(self.mu,self.sd))))
    
    def dist(self,N,bins):
        dist = np.round(np.exp(np.random.normal(self.mu,self.sd,N)))
        plt.hist(dist,bins=bins)
        return dist.mean(), dist.std()
class Poisson:
    def __init__(self,mu):
        self.mu = mu
    def random(self):
        return poisson(self.mu)

class negbin:
    def __init__(self,mu,k):
        self.mu = mu
        self.k = k
    def convert_params(self,mu,theta):
        r = theta
        var = mu + 1 / r * mu ** 2
        p = (var - mu) / var
        return r, 1 - p
    def pmf(self,counts, mu, theta):
        return nbinom.pmf(counts, *self.convert_params(mu, theta))
    def negbin_offspring(self,mu,k):
        v = np.arange(0,40)
        p = np.array(list(map(lambda v: self.pmf(v,mu,k),v)))

        return p
    def random(self):
        dist = self.negbin_offspring(self.mu,self.k)
        cum_dist = pd.Series(dist.cumsum())
        rand = np.random.rand()
        offspringsSr = (cum_dist[rand < cum_dist])
        
        if len(offspringsSr) == 0:
            offsprings = cum_dist.index[-1] + 1
        else:
            offsprings = offspringsSr.index[0]
        return offsprings
    def dist(self):
        dist = self.negbin_offspring(self.mu,self.k)
        cum = dist.cumsum()
        fig,ax = plt.subplots(figsize=(10,5))
        #ax.plot(cum,color='red')
        ax.bar(x=np.arange(len(dist)),height=dist)
        return dist
    def mean(self):
        dist = self.negbin_offspring(self.mu,self.k)
        return (np.arange(len(dist))*dist).sum()


class Infected:
    mean_si = 4
    SI = Poisson(mean_si)
    __id = 1
    def __init__(self,r0,k):
        self.distparams = (r0,k)
        self.r0 = negbin(r0,k)
        self.n_offspring_ = self.r0.random()
        self.onset_ = Infected.SI.random()
        self.__id = Infected.__id
        Infected.__id += 1
    def get_id(self):
        return self.__id
    def set_parent_id(self,parent_id):
        self.__parent_id = parent_id
    def get_parent_id(self):
        return self.__parent_id
    def get_mean_r0(self):
        return self.r0.mean(10000,20)
    def get_si_dist(self):
        return Infected.SI.dist()

    def generate_offspring(self):
        offsprings = []
        for i_offspring in range(self.n_offspring_):
            offspring = Infected(*self.distparams)
            parent_id = self.get_id()
            offspring.set_parent_id(parent_id)
            offsprings.append(offspring)
        return offsprings


if __name__ == "__main__":
    from tqdm import tqdm
    I0 = Infected(2.2,0.2) # Index case
    gens = [I0.generate_offspring()] # First generation

    for k in tqdm(range(5)): # Second generate to sixth generation
        i = -1
        gen = []
        for ind in gens[i]:
            gen = gen + ind.generate_offspring()
        gens.append(gen)
