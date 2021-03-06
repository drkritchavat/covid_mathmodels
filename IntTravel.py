import numpy as np
from scipy.integrate import odeint
from .MathModels import MathModels

class IntTravel:
    def __init__(
        self,
        T = 90,
        y = np.array([69000000,0,0,0,0]),
        R0_I = 1.8,
        R0_Q = 0,
        epsilon_I = 0.2,
        epsilon_E = 0.01,
        dInc = 5.2,
        dInf = 2.3,
        Imps = dict(
            S = 0,
            E = 0.6,
            I = 0.4,
            Q = 0,
            R = 0
        ),
        nImp = 1):
        self.T = T
        self.y = y
        self.t = np.linspace(0,T,T)
        self.R0_I = R0_I
        self.R0_Q = R0_Q
        self.epsilon_I = epsilon_I
        self.epsilon_E = epsilon_E
        self.dInc = dInc
        self.dInf = dInf
        self.Imps = Imps
        self.nImp = nImp
        
    
    def seiqr_imp(
        self,
        y,
        t,
        alpha,
        beta_I,
        beta_Q,
        epsilon_E,
        epsilon_I,
        gamma,
        Imps,
        ):

        """
        Original SEIQR with
        modification of imported cases
        """
        S,E,I,Q,R = y
        Imp_S, Imp_E, Imp_I, Imp_Q, Imp_R = Imps

        dSdt = Imp_S - beta_I*S*I - beta_Q*S*Q 
        dEdt = Imp_E + beta_I*S*I + beta_Q*S*Q - alpha*E
        dIdt = Imp_I + (1-epsilon_E)*alpha*E - epsilon_I*I - gamma*I
        dQdt = Imp_Q + epsilon_E*alpha*E + epsilon_I*I - gamma*Q
        dRdt = Imp_R + gamma*(I+Q)

        dYdt = [
            dSdt,
            dEdt,
            dIdt,
            dQdt,
            dRdt
        ]
        return dYdt
    
    def get_args(self):
        args = dict(
            T = self.T,
            y = self.y,
            R0_I = self.R0_I,
            R0_Q = self.R0_Q,
            epsilon_I = self.epsilon_I,
            epsilon_E = self.epsilon_E,
            dInc = self.dInc,
            dInf = self.dInf,
            Imps = self.Imps,
            nImp = self.nImp
        )
        return args
    
    def run(self):
        args = dict(
            alpha = 1/self.dInc,
            beta_I = self.R0_I/self.y[0]/self.dInf,
            beta_Q = self.R0_Q,
            epsilon_E = self.epsilon_E,
            epsilon_I = self.epsilon_I,
            gamma = 1/self.dInf,
            Imps = np.array([self.Imps[i] for i in self.Imps]) * self.nImp
        )
        y = odeint(
            self.seiqr_imp,
            self.y,
            self.t,
            tuple(args.values())
        )
        return y



