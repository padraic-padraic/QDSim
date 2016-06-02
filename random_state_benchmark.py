from matplotlib import pyplot as plt
from QDSim.sim import Simulation
from QDSim.dispatcher import repeat_execution, star_execution
from QDSim.noise import thermal_state
from QDSim.physics import root_iSWAP
from QDSim.solvers import do_qt_mesolve
from random import random

import qutip as qt
import numpy as np

def random_benchmark(state, **kwargs):
    target = root_iSWAP * state.ptrace([1,2])
    sim = Simulation(do_qt_mesolve, state=state, fname='time_dependent.yaml')
    states = sim.run_solver(nsteps=5000, steps=10000, tau=1e-9, rhs_reuse=True)
    return np.max([qt.fidelity(target, s.ptrace([1,2])) for s in states])

if __name__ == '__main__':
    n = thermal_state(5e9, 20e-3, 5)
    targets = [[qt.tensor(n, qt.rand_ket(2), qt.rand_ket(2))] for i in range(100)]
    kwarg_list = [{}]*100
    results = star_execution(random_benchmark, targets, kwarg_list)
    plt.hist(results,50)
    plt.show()
