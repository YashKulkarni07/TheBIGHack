
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SIRModel:
    def __init__(self, total_pop, infected, recovered, timepoints):
       self.N = total_pop
       self.I0 = infected
       self.R0 = recovered
       self.S0 = self.N - self.I0 - self.R0
       self.beta = 0.2
       self.gamma = 1./10
       self.timepoints = timepoints

    def setBetaGamma(beta, gamma):
        self.beta = beta
        self.gamma = gamma
    # The SIR model differential equations.
    def deriv(self,y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def run(self):
        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, self.timepoints, args=(self.N, self.beta, self.gamma))
        S, I, R = ret.T
        return S, I, R

    def plot(self,S, I, R ):
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(self.timepoints, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(self.timepoints, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(self.timepoints, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number (1000s)')
        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()


