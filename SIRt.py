import numpy as np

from scipy.integrate import odeint
from .MathModels import MathModels

class SIR(MathModels):
    def SIReq(
        self,
        Y,
        t,
        gamma,
        ):
        R0 = self.get_R0(t)
        N = sum(Y)
        dInf = 1/gamma
        beta = self.to_beta(R0=R0,N=N,dInf=dInf)

        """
        Original SIR Model 
        """
        S,I,R = Y

        dY = [
            -beta*S*I,
             beta*S*I - gamma*I,
             gamma*I
        ]
        return dY
    
    def run(self):
        Y = self.get_Y()
        Y = dict(
            S = Y['S'],
            I = Y['I'],
            R = Y['R']
        )
        params = self.get_params()

        kwargs = dict(
            gamma=1/params['dInf']
        )
        result = self.numerical_solve(
            self.SIReq,
            np.array(list(Y.values())),
            self.t,
            tuple(kwargs.values())
        )
        return result
