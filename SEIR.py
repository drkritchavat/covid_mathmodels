import numpy as np
from scipy.integrate import odeint
from MathModels import MathModels

class SEIR(MathModels):
    def SEIReq(
        self,
        Y,
        t,
        alpha,
        beta,
        gamma,
        ):

        """
        Original SEIR Model 
        """
        S,E,I,R = Y

        dY = [
            -beta*S*I,
             beta*S*I - alpha*E,
             alpha*E - gamma*I,
             gamma*I
        ]
        return dY
    
    def run(self):
        Y = self.get_Y()
        Y = dict(
            S = Y['S'],
            E = Y['E'],
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
            alpha = 1/params['dInc'],
            beta = beta,
            gamma = 1/params['dInf']
        )

        result = self.numerical_solve(
            self.SEIReq,
            np.array(list(Y.values())),
            self.t,
            tuple(kwargs.values())
        )
        return result