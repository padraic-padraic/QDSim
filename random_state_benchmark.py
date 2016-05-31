from QDSim import np, Simulation, qt
from QDSim.noise import thermal_state
from QDSim.solvers import do_qt_mesolve
from random import random


def random_benchmark():
    sim = Simulation(do_qt_mesolve,fname='time_dependent.yaml')
    n = thermal_state(sim.w_c, 20e-3, sim.cav_dim)
    q_1 = qt.rand_ket(2)
    q_2 = qt.rand_ket(2)
    sim.set_state(qt.tensor(n, q_1, q_2))
    return sim.run_solver(nsteps=1000, steps=2500, tau=1e-8, progress_bar=True)

if __name__ == '__main__':
    