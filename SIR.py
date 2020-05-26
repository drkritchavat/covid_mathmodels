import numpy as np
from scipy.integrate import odeint
from .MathModels import MathModels

class SIR(MathModels):
    def SIReq(
        self,
        Y,
        t,
        beta,
        gamma,
        ):

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
        beta = self.to_beta(
            R0 = params['R0'],
            N = sum(self.Y.values()),
            dInf = params['dInf']
            )
        kwargs = dict(
            beta=beta,
            gamma=1/params['dInf']
        )
        result = self.numerical_solve(
            self.SIReq,
            np.array(list(Y.values())),
            self.t,
            tuple(kwargs.values())
        )
        return result
