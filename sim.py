from matplotlib import pyplot as plt
from QDSim import *
from QDSim.conf_loader import Conf
from QDSim.dispatcher import repeat_execution, star_execution

import os

class Simulation(Conf):
    def __init__(self, solver, **kwargs):
        self.solver = solver
        super().__init__(kwargs.pop('fname','test.yaml'))
        state = kwargs.pop('state',None)
        if state:
            self.state = state
            self.dims = state.dims[0]
        if kwargs.pop('H',None):
            self.H = H
        self.lindblads = kwargs.pop('lindblads',[])

    def set_state(self, _state):
        self.state = _state
        self.dims = _state.dims[0]

    def set_H(self,_H):
        self.H = _H

    def append_L(self,_lindblad):
        self.lindblads.append(_lindblad)

    def run_solver(self,*args, **kwargs):
        steps = kwargs.pop('steps', 3000)
        tau = kwargs.pop('tau', .5e-9)
        print(steps,tau)
        return self.solver(self.state, self.H, self.lindblads,
                                             steps, tau, **kwargs)
    def parse_states(self,results):
        steps = len(results)
        qubit_indices,cav_index = parse_dims(np.array(self.dims))
        cavity_sim = cav_index is not None
        if cavity_sim:
            n = qt.num(self.dims[cav_index])
        bloch_vectors = np.zeros((steps,len(qubit_indices),3),dtype=np.complex_)
        ns = np.zeros(steps,dtype=np.complex_)
        print(qubit_indices,cav_index)
        for i in range(steps):
            if results[i].type != 'oper':
                dm = qt.ket2dm(results[i])
            else:
                dm = results[i]
            bloch_vectors[i] = [measure_qubit(dm.ptrace(int(j))) for j in qubit_indices]
            if cavity_sim:
                ns[i] = (n*dm.ptrace(cav_index)).tr()
        return bloch_vectors,ns

    def get_final_only(self, **kwargs):
        steps = kwargs.pop('steps',3000)
        tau = kwargs.pop('steps',.5e-9)
        return (self.run_solver(kwargs))[-1]

    def average_trajectories(self,n=100,**kwargs):
        if sim.solver.__name__ == 'do_qt_mcsolve':
            return self.run_solver(ntraj=n)
        else:
            res = repeat_execution(n, self.run_solver, [], kwargs)
            bloch_vectors = np.array([el[0] for el in res]).mean(0)
            ns = np.array([el[1] for el in res]).mean(0)
            return bloch_vectors,ns

    def iter_params(self,arg_list,kwarg_list):
        res = star_execution(self.get_final_only,arg_list,kwarg_list)
        return


    def make_plots(self,bvs,ns):
        b = qt.Bloch()
        b.point_color = ['r','g','b']
        # b.point_marker = ['o','s','d','^']
        b.add_points([bvs[:,0,0], bvs[:,0,1], bvs[:,0,2]],'l')
        b.add_points([bvs[:,1,0], bvs[:,1,1], bvs[:,1,2]],'l')
        b.show()
        if np.any(ns):
            plt.hist(np.real(ns), 10)
            plt.show()

fname = os.path.join(os.path.dirname(__file__),'test.yaml')
# print(fname)
# cav = qt.Qobj(np.sqrt((qt.num(5)*qt.thermal_dm(5,0.04)).diag()))
cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = (qt.basis(2,1)+qt.basis(2,0)).unit()
L1 = 0.01 * qt.tensor(qt.qeye(5),qt.destroy(2),I)
L2 = 0.01 * qt.tensor(qt.qeye(5),I,qt.destroy(2))
# lindblads = [L1,L2]

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
    states = []
    sim = Simulation(do_qt_mesolve, state=qt.tensor(q1,q2),fname='2qtest.yaml')#, lindblads=lindblads)
    arg_list = [[]]*3000
    kwarg_list = [{'steps':i,'nsteps':2500} for i in range(1,3001)]
    states = sim.iter_params(arg_list,kwarg_list)
    sim.make_plots(*sim.parse_states(states))
    # sim = Simulation(do_qt_mesolve, state=qt.tensor(q1,q2),fname='2qtest.yaml')#, lindblads=lindblads)
    # sim.set_state(sim.iSWAP_U*sim.state)
    # states = sim.run_solver(nsteps=2500)
    sim.make_plots(*sim.parse_states(states))
    # sim = Simulation(do_qt_mesolve,state=qt.tensor(cav,q1,q2))
    # states= sim.run_solver(nsteps=2500)
    # sim.make_plots(*sim.parse_states(states))
