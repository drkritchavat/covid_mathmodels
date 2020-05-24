import numpy as np
from scipy.integrate import odeint
from MathModels import MathModels

class SEIQR(MathModels):
    def SEIQReq(
        self,
        Y,
        t,
        alpha,
        beta_I,
        beta_Q,
        epsilon_E,
        epsilon_I,
        gamma,
        ):

        """
        Original SEIR Model 
        """
        S,E,I,Q,R = Y

        dY = [
             -beta_I*S*I - beta_Q*S*Q,
             beta_I*S*I + beta_Q*S*Q - alpha*E,
             (1-epsilon_E)*alpha*E - epsilon_I*I - gamma*I,
             epsilon_E*alpha*E + epsilon_I*I - gamma*Q,
             gamma*(I+Q)
        ]
        return dY
    
    def run(self):
        Y = self.get_Y()
        Y = dict(
            S = Y['S'],
            E = Y['E'],
            I = Y['I'],
            Q = Y['S'],
            R = Y['R'],
        )
        params = self.get_params()
        beta_I = self.to_beta(
            R0 = params['R0_I'],
            N = sum(self.Y.values()),
            dInf = params['dInf'],
            )
        beta_Q = self.to_beta(
            R0 = params['R0_Q'],
            N = sum(self.Y.values()),
            dInf = params['dInf'],
            )
        kwargs = dict(
            alpha = 1/params['dInc'],
            beta_I = beta_I,
            beta_Q = beta_Q,
            epsilon_E = params['epsilon_E'],
            epsilon_I = params['epsilon_I'],
            gamma = 1/params['dInf'],
        )

        result = self.numerical_solve(
            self.SEIQReq,
            np.array(list(Y.values())),
            self.t,
            tuple(kwargs.values())
        )
        return result