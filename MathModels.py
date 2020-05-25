import numpy as np
from scipy.integrate import odeint
import pandas as pd
class MathModels:
    def __init__(self,Y,T,params):
        self.params = params
        self.Y = Y
        self.t = np.linspace(0,T,T)
    def get_params(self):
        return self.params
    def get_Y(self):
        return self.Y
    def set_params(self,params):
        self.params = params
    def set_Y(self,Y):
        self.Y = Y
    def set_T(self,T):
        self.t = np.linspace(0,T,T)
    def add_params(self,params):
        self.params.update(params)
    def add_Y(self,Y):
        self.Y.update(Y)
    def to_beta(self,R0,N,dInf):
        beta = R0/N/dInf
        return beta

    def set_R0t(self,R0t):
        """
        Take Pandas dataframe with R0 and time
        
        """
        self.R0t = R0t

    def get_R0(self,t):
        query = self.R0t.query("t >= @t")
        if len(query) == 0:
            R0 = self.R0t['R0'].values[-1]
        else:
            R0 = query['R0'].values[0]
        return R0

    def numerical_solve(self,eq,Y,t,args):
        result = odeint(
            eq,
            Y,
            t,
            args
            )
        self.result = result
        return result
    def actual_data(self,data):
        self.data = data
    def RMSE(self,metric):
        actual = np.array(self.data)
        predicted = np.array(self.result)[:,metric]
        RMSE = np.sqrt(((actual-predicted)**2).mean())
        return RMSE