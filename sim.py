from matplotlib import pyplot as plt
from QDSim import *
from QDSim.conf_loader import load

import os

class Simulation():
    def __init__(self, solver, state=None,H=None,lindblads=[]):
        self.solver = solver
        if state:
            self.state = state
        if H:
            self.H = H
        self.lindblads = lindblads

    def set_state(self,_state):
        self.state = _state

    def set_H(self,_H):
        self.H = _H

    def append_L(self,_lindblad):
        self.lindblads.append(_lindblad)

    def run_solver(self,**kwargs):
        steps = kwargs.get('steps',1000)
        tau = kwargs.get('tau',1./100)
        return self.solver(self.state, self.H, self.lindblads,
                                             steps,tau)


fname = os.path.join(os.path.dirname(__file__),'test.yaml')
# print(fname)
H = load(fname)
cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = qt.basis(2,0)

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
    # bloch_vectors,n = test_mc_sim()
    # b = qt.Bloch()
    # b.add_points(bloch_vectors[0,:,:])
    # # b.add_points([x2,y2,z2])
    # b.show()
    # print(n)
    # plt.hist(np.real(n),10)
    # plt.show()
    # for i in range(x1.size):
    #     b.clear()
    #     b.add_points([x1[:i+1],y1[:i+1],z1[:i+1]])
    #     b.save(dirc='temp')
    # import timeit
    # print(timeit.Timer("load_H()",setup="from __main__ import load_H").repeat(3,1))
    # print(timeit.Timer("test_mc_sim()",setup="from __main__ import test_mc_sim").repeat(3,1))
    # print(timeit.Timer("test_rk4_sim()",setup="from __main__ import test_rk4_sim").repeat(3,1))
    from QDSim.dispatcher import repeat_execution
    sim = Simulation(do_jump_mc,qt.tensor(cav,q1,q2),H)
    res = repeat_execution(sim.run_solver,[],1)
    print(type(res))
    print(res[0][0].shape)