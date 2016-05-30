from copy import deepcopy
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
        else:
            self._load_H()
        self.lindblads = kwargs.pop('lindblads',[])

    def set_state(self, _state):
        self.state = _state
        self.dims = _state.dims[0]

    def set_H(self,_H):
        self.H = _H

    def append_L(self,_lindblad):
        self.lindblads.append(_lindblad)

    def run_solver(self,*args, **kwargs):
        steps = kwargs.pop('steps', 1000)
        tau = kwargs.pop('tau', 1e-9)
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
        return (self.run_solver(**kwargs))[-1]

    def average_trajectories(self,n=100,**kwargs):
        if sim.solver.__name__ == 'do_qt_mcsolve':
            return self.run_solver(ntraj=n,**kwargs)
        else:
            res = repeat_execution(n, self.run_solver, [], kwargs)
            bloch_vectors = np.array([el[0] for el in res]).mean(0)
            ns = np.array([el[1] for el in res]).mean(0)
            return bloch_vectors,ns

    def iter_params(self,f,arg_list,kwarg_list):
        return star_execution(f,arg_list,kwarg_list)

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

# cav = qt.Qobj(np.sqrt((qt.num(5)*qt.thermal_dm(5,0.04)).diag()))
cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = (qt.basis(2,1))#+qt.basis(2,0)).unit()
# L1 = 0.01 * qt.tensor(qt.qeye(5),qt.destroy(2),I)
# L2 = 0.01 * qt.tensor(qt.qeye(5),I,qt.destroy(2))
# lindblads = [L1,L2]

if __name__ == '__main__':
    states = []
    sim = Simulation(do_qt_mcsolve,state=qt.tensor(cav,q1,q2),fname='test.yaml')
    # arg_list = [[]]*4
    # kwarg_list = []
    # for i in range(4):
    #     kwarg_list.append({'nsteps':10000, 'steps':3500, 'tau':1e-6,
    #                        'times':np.linspace(3500*i, (3500*i)+3500, 3500)})
    states = sim.run_solver(steps=140000,tau=1e-7,nsteps=5000)
    target = root_iSWAP*qt.tensor(q1,q2)
    fids = np.array([qt.fidelity(target,state.ptrace([1,2])) for state in states])
    print(np.max(fids),np.argmax(fids))
    bvs,ns = sim.parse_states(states)
    with open('res2.txt','w') as f:
        for i, fid in enumerate(fids):
            f.write(str(1e-7*i) + "\t" + str(fid)+"\t"+ str(ns[i]) + "\t"
                    + str(bvs[i,0,0]) + "\t" + str(bvs[i,0,1]) + "\t"
                    + str(bvs[i,0,2]) + "\t" + str(bvs[i,1,0]) + "\t"
                    + str(bvs[i,1,1]) + "\t" + str(bvs[i,1,2]) + "\n")
    # print(sim.run_solver(nsteps=2500,steps=500,tau=1e-5))
    # taus = [1e-7,5e-7,1e-6,5e-6]
    # for tau in taus:
    #     sim = Simulation(do_qt_mesolve, state=qt.tensor(q1, q2), fname='2qtest.yaml')
    #     sim._load_H()
    #     arg_list = [[]]*1000
        # kwarg_list = [{'steps':i,'tau':tau} for i in range(1,1001)]
    #     states = sim.iter_params(arg_list,kwarg_list)
    #     # states = [qt.ket2dm(state).ptrace([1,2]) for state in states]
    #     target = iSWAP*qt.tensor(q1,q2)
    #     fids = np.array([qt.fidelity(target,state) for state in states])
    #     print(np.max(fids),np.argmax(fids))
    #     with open(str(tau)+'.txt','w') as f:
    #         for i, fid in enumerate(fids):
    #             f.write(str((i+1)*tau) + " \t " + str(fid) + "\n")
    #     print('Done for tau ' + str(tau))
