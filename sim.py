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
        steps = kwargs.get('steps', 10000)
        tau = kwargs.get('tau', 1.)
        return self.solver(self.state, self.H, self.lindblads,
                                             steps, tau)

    def average_trajectories(self,n=100,**kwargs):
        res = repeat_execution(n, self.run_solver, [], kwargs)
        bloch_vectors = np.array([el[0] for el in res]).mean(0)
        ns = np.array([el[1] for el in res]).mean(0)
        return bloch_vectors,ns

    def make_plots(self,bvs,ns):
        b = qt.Bloch()
        # b.add_points([bvs[:,0,0], bvs[:,0,1], bvs[:,0,2]])
        b.add_points([bvs[:,1,0], bvs[:,1,1], bvs[:,1,2]])
        b.show()
        plt.hist(np.real(ns), 10)
        plt.show()

fname = os.path.join(os.path.dirname(__file__),'test.yaml')
# print(fname)
H = load(fname)
cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = (qt.basis(2,1)+qt.basis(2,0)).unit()

# lindblads = [L1,L2]

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
    repeat_execution(sim.run_solver,[],5)

if __name__ == '__main__':
    # sim = Simulation(do_jump_mc,qt.tensor(cav,q1,q2),H)
    # bvs,ns = sim.average_trajectories(10)
    sim = Simulation(do_rk4,qt.tensor(cav,q1,q2),H)
    bvs,ns = sim.run_solver()
    sim.make_plots(bvs,ns)