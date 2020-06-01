import numpy as np
from scipy.integrate import odeint
from MathModels import MathModels

class SEIQRInt(MathModels):
    def SEIQReq(
        self,
        Y,
        t,
        alpha,
        beta_I,
        beta_Q,
        beta_Int,
        epsilon_E,
        epsilon_I,
        epsilon_EInt,
        epsilon_IInt,
        Imp_E,
        Imp_I,
        gamma,
        ):

        """
        Original SEIR Model 
        """
        S,E,I,Q,R,E_Int,I_Int,Q_Int,R_Int = Y

        dY = [
             -beta_I*S*I - beta_Q*S*Q - beta_Int*S*I_Int,
             beta_I*S*I + beta_Q*S*Q - alpha*E + beta_Int*S*I_Int,
             (1-epsilon_E)*alpha*E - epsilon_I*I - gamma*I,
             epsilon_E*alpha*E + epsilon_I*I - gamma*Q,
             gamma*(I+Q),
             Imp_E - alpha*E_Int,
             Imp_I + (1-epsilon_EInt)*alpha*E_Int - epsilon_IInt*I_Int - gamma*I_Int,
             epsilon_EInt*alpha*E_Int + epsilon_IInt*I_Int - gamma*Q_Int,
             gamma*(I_Int+Q_Int)
        ]
        return dY
    
    def run(self):
        Y = self.get_Y()
        Y = dict(
            S = Y['S'],
            E = Y['E'],
            I = Y['I'],
            Q = Y['Q'],
            R = Y['R'],
            E_Int = Y['E_Int'],
            I_Int = Y['I_Int'],
            Q_Int = Y['Q_Int'],
            R_Int = Y['R_Int']
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
        beta_Int = self.to_beta(
            R0 = params['R0_Int'],
            N = sum(self.Y.values()),
            dInf = params['dInf'],
            )
        kwargs = dict(
            alpha = 1/params['dInc'],
            beta_I = beta_I,
            beta_Q = beta_Q,
            beta_Int = beta_Int,
            epsilon_E = params['epsilon_E'],
            epsilon_I = params['epsilon_I'],
            epsilon_EInt = params['epsilon_EInt'],
            epsilon_IInt = params['epsilon_IInt'],
            Imp_E = params['Imp_E'],
            Imp_I = params['Imp_I'],
            gamma = 1/params['dInf'],
        )

        result = self.numerical_solve(
            self.SEIQReq,
            np.array(list(Y.values())),
            self.t,
            tuple(kwargs.values())
        )
        return result
