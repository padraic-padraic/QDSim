from matplotlib import pyplot as plt
from QDSim.conf_loader import Conf
from QDSim.dispatcher import repeat_execution, star_execution
from QDSim.noise import *
from QDSim.physics import * 
from QDSim.solvers import *

import qutip as qt
import numpy as np

__all__ = ["Simulaiton"]

class Simulation(Conf):
    def __init__(self, solver, **kwargs):
        self.solver = solver
        self.dims = None
        super().__init__(kwargs.pop('fname','test.yaml'))
        state = kwargs.pop('state',None)
        if state:
            self.set_state(state)
        if kwargs.pop('H',None):
            self.H = H
            if self.dims is None:
                self.dims = H.dims[0]
        else:
            self._load_H()
        self.lindblads = kwargs.pop('lindblads',[])

    def set_state(self, _state):
        self.state = _state
        self.dims = np.array(_state.dims[0])

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
        qubit_indices,cav_index = parse_dims(self.dims)
        cavity_sim = cav_index is not None
        if cavity_sim:
            n = qt.num(self.dims[cav_index])
        bloch_vectors = np.zeros((steps,len(qubit_indices),3))
        ns = np.zeros(steps)
        for i in range(steps):
            if results[i].type != 'oper':
                dm = qt.ket2dm(results[i])
            else:
                dm = results[i]
            bloch_vectors[i] = [measure_qubit(dm.ptrace(int(j))) for j in qubit_indices]
            if cavity_sim:
                ns[i] = np.real((n*dm.ptrace(cav_index)).tr())
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


    def to_file(self, tau, states, outname, fidelities=[]):
        bvs, ns = self.parse_states(states)
        with open(outname+'.txt', 'w') as f:
            for i in range(ns.size):
                f.write(str(tau*i) + "\t" + 
                        str(bvs[i, 0, 0]) + "\t" + 
                        str(bvs[i, 0, 1]) + "\t" + 
                        str(bvs[i, 0, 2]) + "\t" +
                        str(bvs[i, 0, 0]) + "\t" + 
                        str(bvs[i, 0, 1]) + "\t" + 
                        str(bvs[i, 0, 2]) + "\t" +                        
                        str(ns[i]) + "\n"
                    )
        if fidelities != []:
            with open(outname+'_fids'+'.txt', 'w') as f:
                for i in range(len(fidelities)):
                    f.write(str(tau*i) + "\t" + str(fidelities[i]) + "\n")

# cav = qt.Qobj(np.sqrt((qt.num(5)*qt.thermal_dm(5,0.04)).diag()))
cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = (qt.basis(2,1)+qt.basis(2,0)).unit()
# L1 = 0.01 * qt.tensor(qt.qeye(5),qt.destroy(2),I)
# L2 = 0.01 * qt.tensor(qt.qeye(5),I,qt.destroy(2))
# lindblads = [L1,L2]

if __name__ == '__main__':
    # cav = thermal_state(5e9,20e-3)
    sim = Simulation(do_qt_mesolve, state=qt.tensor(cav, q1, q2),
                     fname='time_dependent.yaml')
    # sim.append_L(cavity_loss(10000, sim.w_c, 20e-3, sim.dims))
    # sim.append_L(thermal_in(10000, sim.w_c, 20e-3, sim.dims))
    # sim.append_L(relaxation(1,sim.dims,1))
    # sim.append_L(relaxation(1,sim.dims,2))
    # sim.append_L(decoherence(1e2,sim.dims,1))
    # sim.append_L(decoherence(1e2,sim.dims,2))
    states = sim.run_solver(nsteps=1000, steps=10000, tau=1e-9,
                            progress_bar=True)
    target = root_iSWAP * qt.tensor(q1, q2)
    fids1 = [qt.fidelity(target, state.ptrace([1,2])) for state in states]
    # print(np.max(fids1),np.argmax(fids1)*1e-9)
    # sim = Simulation(do_qt_mesolve, state=qt.tensor(q1, q2),
    #                  fname='2qtest.yaml')
    # states = sim.run_solver(nsteps=1000,steps=2500,tau=1e-7,progress_bar=True)
    # target = root_iSWAP * qt.tensor(q1, q2)
    # fids2 = [qt.fidelity(target, state) for state in states]
    # print(np.max(fids1),np.argmax(fids1)*1e-7)
    plt.plot(1e-7*np.arange(len(fids1)), fids1, 'r')
    #          # 1e-7*np.arange(len(fids2)), fids2, 'b')
    plt.show()
