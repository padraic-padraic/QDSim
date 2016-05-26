from matplotlib import pyplot as plt
from QDSim import *
from QDSim.conf_loader import load
from QDSim.dispatcher import repeat_execution

import os

class Simulation():
    def __init__(self, solver, state=None, H=None, lindblads=[]):
        self.solver = solver
        if state:
            self.state = state
        if H:
            self.H = H
        self.lindblads = lindblads

    def set_state(self, _state):
        self.state = _state

    def set_H(self,_H):
        self.H = _H

    def append_L(self,_lindblad):
        self.lindblads.append(_lindblad)

    def run_solver(self,**kwargs):
        steps = kwargs.pop('steps', 1000000)
        tau = kwargs.pop('tau', 1e-15)
        print(steps,tau)
        return self.solver(self.state, self.H, self.lindblads,
                                             steps, tau, **kwargs)

    def average_trajectories(self,n=100,**kwargs):
        if sim.solver.__name__ == 'do_qt_mcsolve':
            return self.run_solver(ntraj=n)
        else:
            res = repeat_execution(n, self.run_solver, [], kwargs)
            bloch_vectors = np.array([el[0] for el in res]).mean(0)
            ns = np.array([el[1] for el in res]).mean(0)
        return bloch_vectors,ns

    def make_plots(self,bvs,ns):
        b = qt.Bloch()
        b.point_color = ['r','g','b']
        # b.point_marker = ['o','s','d','^']
        b.add_points([bvs[:,0,0], bvs[:,0,1], bvs[:,0,2]])
        b.add_points([bvs[:,1,0], bvs[:,1,1], bvs[:,1,2]])
        b.show()
        plt.hist(np.real(ns), 10)
        plt.show()

fname = os.path.join(os.path.dirname(__file__),'test.yaml')
# print(fname)
H = load(fname)
cav = qt.Qobj(np.sqrt((qt.num(5)*qt.thermal_dm(5,0.04)).diag()))
# cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = (qt.basis(2,1)+qt.basis(2,0)).unit()
L1 = 1e5 * qt.tensor(qt.qeye(5),qt.destroy(2),I)
L2 = 1e5 * qt.tensor(qt.qeye(5),I,qt.destroy(2))
lindblads = [L1,L2]

def load_H():
    H = load(fname)

def test_mc_sim():
    sim = Simulation(do_jump_mc,qt.tensor(cav,q1,q2),H)
    return sim.run_solver()

def test_rk4_sim():
    sim = Simulation(do_rk4,qt.tensor(cav,q1,q2),H)
    sim.run_solver()

def test_parallel():
    sim = Simulation(do_jump_mc,qt.tensor(cav,q1,q2),H)
    sim.average_trajectories(5)

if __name__ == '__main__':
    # sim = Simulation(do_jump_mc,qt.tensor(cav,q1,q2),H)
    # bvs,ns = sim.average_trajectories(10)
    # import timeit
    # print(np.sum(timeit.Timer('test_mc_sim()',setup='from __main__ import test_mc_sim').repeat(5,1)))
    # print(np.sum(timeit.Timer('test_rk4_sim()',setup='from __main__ import test_rk4_sim').repeat(5,1)))
    # print(timeit.Timer('test_parallel()', setup='from __main__ import test_parallel').repeat(1,1))
    sim = Simulation(do_qt_mcsolve,qt.tensor(cav,q1,q2),H,lindblads)
    bvs,ns = sim.average_trajectories(500)
    sim.make_plots(bvs,ns)