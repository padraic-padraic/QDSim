from matplotlib import pyplot as plt
from QDSim.sim import Simulation
from QDSim.dispatcher import repeat_execution, star_execution
from QDSim.noise import thermal_state, get_lind_list
from QDSim.physics import root_iSWAP
from QDSim.solvers import do_qt_mesolve
from random import random

import qutip as qt
import numpy as np

def random_benchmark(state, **kwargs):
    target = root_iSWAP * state#.ptrace([1,2])
    sim = Simulation(do_qt_mesolve, state=state, fname='2qtest.yaml')
    states = sim.run_solver(nsteps=1000, steps=1000, tau=1e-8,
                            progress_bar=True)
    return np.max([qt.fidelity(target, s) for s in states])

def random_purity_benchmark(state, **kwargs):
    target = root_iSWAP * state.ptrace([1,2])
    sim = Simulation(do_qt_mesolve, state=state, fname='time_dependent.yaml')
    states = sim.run_solver(nsteps=2000, steps=10000, tau=1e-7, rhs_reuse=True)
    print('Done a state')
    return qt.fidelity(target, states[-1].ptrace([1,2]))


if __name__ == '__main__':
    n = thermal_state(5e9, 20e-3, 5)
    state = qt.tensor(n, qt.rand_ket(2), qt.rand_ket(2))
    sim = Simulation(do_qt_mesolve, state=state, fname='time_dependent.yaml')
    ls = get_lind_list(sim.w_c, 1e4, 20e-3, 1, 1e2, sim.dims)
    sim.lindblads = ls
    res = sim.run_solver(nsteps=1000, steps=10000, tau=1e-7, progress_bar=True)
    # bvs, ns = sim.parse_states(res)
    b = qt.Bloch()
    for i in range(10000):
        b.clear()
        b.add_states(res[i].ptrace(1))
        b.add_states(res[i].ptrace(2))
        b.render()
        b.save(dirc='temp')
    # plt.hist(ns, 50)
    # plt.show()
    # targets = [[qt.tensor(qt.rand_ket(2), qt.rand_ket(2))] for i in range(100)]
    # kwarg_list = [{}]*100
    # results = star_execution(random_benchmark, targets, kwarg_list)
    # plt.hist(results,50)
    # plt.show()
