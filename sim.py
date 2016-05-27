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
        tau = kwargs.pop('tau', .1e-10)
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
        return (self.run_solver(**kwargs))[-1]

    def average_trajectories(self,n=100,**kwargs):
        if sim.solver.__name__ == 'do_qt_mcsolve':
            return self.run_solver(ntraj=n,**kwargs)
        else:
            res = repeat_execution(n, self.run_solver, [], kwargs)
            bloch_vectors = np.array([el[0] for el in res]).mean(0)
            ns = np.array([el[1] for el in res]).mean(0)
            return bloch_vectors,ns

    def iter_params(self,arg_list,kwarg_list):
        return star_execution(self.get_final_only,arg_list,kwarg_list)

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

if __name__ == '__main__':
    states = []
    sim = Simulation(do_qt_mesolve, state=qt.tensor(q1, q2), fname='2qtest.yaml')
    sim._load_H()
    arg_list = [[]]*10000
    kwarg_list = [{'steps':i,'nsteps':2500} for i in range(1,10001)]
    states = sim.iter_params(arg_list,kwarg_list)
    target = iSWAP*qt.tensor(q1,q2)
    fids = np.array([qt.fidelity(target,state) for state in states])
    print(np.max(fids),np.argmax(fids))
