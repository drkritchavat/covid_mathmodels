import numpy as np
from scipy.integrate import odeint
class Models:
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

    def numerical_solve(self,eq,Y,t,args):
        result = odeint(
            eq,
            Y,
            t,
            args
            )
        return result

class SIR(Models):
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

class SEIR(Models):
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

class int_travel:
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
