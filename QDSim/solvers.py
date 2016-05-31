"""Module providing interfaces to different dynamics solvers"""
import qutip as qt
from random import random

__all__ = ["parse_dims", "measure_qubit", "do_qt_mcsolve", "do_qt_mesolve",
           "do_jump_mc", "do_rk4"]

def build_S_total(n_spins,operator):
    _dims = [2]*n_spins
    op = qt.Qobj(dims=[_dims,_dims])
    _id = np.identity(n_spins)
    for i in range(n_spins):
        op += qt.tensor([operator if el == 1 else I for el in _id[i,:]])
    return op

def find_jump(r,probabilities):
    """Function used in jump_mc to pick the jump operator"""
    cumsum = np.cumsum(probabilities)
    for n in range(len(cumsum)):
        if r > cumsum[n]:
            return n
    else:
        #Fallback, necessary sometimes, probably due to floating point weirdness
        return None
        print('WHY IS THIS HAPPENING WHY WHY WHY')


def get_prob(dm,L):
    """Get the probability for each jump operator"""
    return (L.dag()*L*dm).tr()

def measure_qubit(dm):
    dm = dm.unit()
    u = (SX*dm).tr()
    v = (SY*dm).tr()
    w = (SZ*dm).tr()
    return u,v,w


def get_M0(H,lindblads,tau):
    _dims = H.dims[0]
    J = qt.Qobj(np.zeros_like(H.data.todense()),dims=[_dims,_dims])
    for Op in lindblads:
        J += Op.dag()*Op
    J *= 0.5
    return qt.tensor([qt.qeye(dim) for dim in _dims]) - tau*(1j*H - J)

def parse_dims(dims):
    qubits = dims == 2
    cav = np.logical_not(qubits)
    indices = np.arange(len(dims))
    if not np.any(cav):
        return indices[qubits],None
    else:
        return list(indices[qubits]),np.asscalar(indices[cav])

def do_qt_mcsolve(state,H,lindblads,steps,tau,**kwargs):
    progress = kwargs.pop('progress_bar',None)
    times = kwargs.pop('times',None)
    if times is None:
        times = np.linspace(0,steps*tau,steps,dtype=np.float_)
    return qt.mcsolve(H,state,times,lindblads,[],
                             options=qt.Options(**kwargs),
                             progress_bar=progress).states
    

def do_qt_mesolve(state,H,lindblads,steps,tau,**kwargs):
    progress = kwargs.pop('progress_bar',None)
    times = kwargs.pop('times',None)
    if times is None:
        times = np.linspace(0,steps*tau,steps,dtype=np.float_)
    return qt.mesolve(H,qt.ket2dm(state),times,lindblads,[],
                             options=qt.Options(**kwargs),
                             progress_bar=progress).states

def do_jump_mc(state,H,lindblads=[],steps=1000,tau=1./10000):
    """Quantum Jump Monte Carlo for simulating Master Equation dynamics. 
        Implemented following the recipe in Harroche & Raimond 
    """
    #Check if our lindblads are time/state dependent
    time_dependent = any([hasattr(Op,'__call__')] for Op in lindblads)
    #Build the regular old evolution operator M0
    M0 = get_M0(H, lindblads,tau)
    states = []
    for i in range(steps):
        states.append(state)
        #Refresh lindblads if time dependent
        if time_dependent:
            #Refresh lindlabds if necessary
            lindblads = [Op(state, i*tau) if type(Op)=='function' else Op for Op in lindblads]
            #Refresh J, 'regular' evolution operator
            M0 = get_M0(H,lindblads,tau)
        p_jump = tau * np.array([get_prob(dm, Op) for Op in lindblads], dtype=np.complex_)
        p_nojump = 1.-np.sum(p_jump)
        r = random()
        if r > p_nojump:
            print('Jump on round ' + str(i))
            #Find the correct jump operator
            index = find_jump(r,p_jump)
            if index is not None:
                state = lindblads[index]*state / np.sqrt(p_jump[index]/tau)
            else:#Fallback option, sometimes necessary due to floating point weirdness
                state = M0*state / np.sqrt(p_nojump)
        else:
            state = M0 * state / np.sqrt(p_nojump)
        state = state.unit()
    return states

def drho(rho,H,lindblads):
    res = -1j* qt.commutator(H,rho)
    if lindblads:
        for l in lindblads:
            res += (l*rho*l.dag() - 0.5* l.dag()*l*rho - 0.5*rho*l.dag()*l)
    return res

def do_rk4_step(rho,H,lindblads,tau,**kwargs):
    k1 = drho(rho,H,lindblads)
    k2 = drho(rho+0.5*tau*k1,H,lindblads)
    k3 = drho(rho+0.5*tau*k2,H,lindblads)
    k4 = drho(rho+tau*k3,H,lindblads)
    return (k1 + 2*k2 + 2*k3 + k4)*tau/6.

def do_rk4(state,H,lindblads=[],steps=1000,tau=1/100,**kwargs):
    states = []
    rho = qt.ket2dm(state)
    for i in range(steps):
        states.append(rho)
        rho = (rho + tau*do_rk4_step(rho,H,lindblads,tau))
        rho = rho.unit()
    return states
