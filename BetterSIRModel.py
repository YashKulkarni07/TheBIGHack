
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Adapted from: https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

class SIRModel:
    # __init__
    # This function inits the SIRModel object
    #
    # preconditions
    #   self - this object
    #   total_pop - the total population
    #   infected - the infected population
    #   recoverd - the recovered population
    #   timepoints - the timepoints that are needed
    def __init__(self, total_pop, infected, dead, timepoints, recovered):
       # Set the parameters that were passed in
       self.N = total_pop
       self.I0 = infected
       self.D0 = dead
       self.R0 = recovered
       self.timepoints = timepoints

       # Suspecible population = total_pop - infected - recovered
       self.S0 = self.N - self.I0 - self.R0

       # Set the infection rate and recovery rate

       self.beta = .2

       self.gamma = 1/10

       # based off of : https://coronavirus.jhu.edu/data/mortality
       self.CFR = .038



       # Delta = the number of people who can get reinfected.
       #  Saying that the antibodies wear off after a 6 months or so
       #  So far nobody knows if this is possible with only a few cases of reinfection: https://www.popsci.com/story/health/covid-19-reinfection-catch-twice/
       self.delta = 1/180


    # The SIR model differential equations.
    def deriv(self,y, t, N, beta, gamma, delta,CFR):
        S, I, R, D = y
        dSdt = -beta * (I * S)/N + delta * R
        dIdt = beta * (I * S)/N - gamma * I
        dRdt = gamma * (1 - CFR) * I - (delta * R)
        dDdt = (gamma * CFR) * I


        #dSdt = -beta * S * I / N
        #dIdt = beta * S * I / N - gamma * I
        #dRdt = gamma * I
        return dSdt, dIdt, dRdt, dDdt

    def run(self):
        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0, self.D0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, self.timepoints, args=(self.N, self.beta, self.gamma, self.delta, self.CFR))
        S, I, R, D = ret.T
        return S, I, R, D

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


