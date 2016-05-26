import qutip as qt
import numpy as np

from random import random
#IDK why qutip doesn't just make these objects?
I = qt.qeye(2)
SX = qt.sigmax()
SY = qt.sigmay()
SZ = qt.sigmaz()
#We're treating '1' as the excited state
SPlus = qt.sigmam()
SMinus = qt.sigmap()
# Do a lindblad-ish thing
# Calculate d-pho/dt for each step totally w/ fine-grained t
# Alternatively, use the MC recipe from harroche

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

def parse_dims(state):
    _dims = np.array(state.dims[0])
    qubits = _dims == 2
    cav = np.logical_not(qubits)
    indices = np.arange(_dims.size)
    if not any(cav):
        return indices[qubits],None
    else:
        return indices[qubits],np.asscalar(indices[cav])

def do_jump_mc(state,H,lindblads=[],steps=1000,tau=1/100):
    """Quantum Jump Monte Carlo for simulating Master Equation dynamics. 
        Implemented following the recipe in Harroche & Raimond 
    """
    #Prebuild arrays
    tau = 1/steps
    qubit_indices,cav_index = parse_dims(state)
    cavity_sim = cav_index is not None
    bloch_vectors = np.zeros((steps,len(qubit_indices),3),dtype=np.complex_)
    n = np.zeros(steps, dtype=np.complex_)
    num = qt.num(state.dims[0][cav_index])
    #Check if our lindblads are time/state dependent
    time_dependent = any([hasattr(Op,'__call__')] for Op in lindblads)
    #Build the regular old evolution operator M0
    M0 = get_M0(H, lindblads,tau)
    for i in range(steps):
        #Refresh lindblads if time dependent
        dm = qt.ket2dm(state)
        if cavity_sim:
            n[i] = (num* dm.ptrace(cav_index)).tr()
        components = [measure_qubit(dm.ptrace(np.asscalar(i))) for i in qubit_indices]
        bloch_vectors[i] = components
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
            state = M0*state / np.sqrt(p_nojump)
        state = state.unit()
    return bloch_vectors,n

def do_genericT2_mc():
    pass

def drho(rho,H,lindblads):
    res = -1j* qt.commutator(H,rho)
    if lindblads:
        for l in lindblads:
            res += (l*rho*l.dag() -0.5* l.dag()*l*rho - 0.5*rho*l.dag()*l)
    return res

def do_rk4_step(rho,H,lindblads,tau):
    k1 = drho(rho,H,lindblads)
    k2 = drho(rho+0.5*tau*k1,H,lindblads)
    k3 = drho(rho+0.5*tau*k2,H,lindblads)
    k4 = drho(rho+tau*k3,H,lindblads)
    return (k1 + 2*k2 + 2*k3 + k4)*tau/6.

def do_rk4(state,H,lindblads=[],steps=1000,tau=1/1000):
    rho = qt.ket2dm(state)
    qubit_indices,cav_index = parse_dims(state)
    bloch_vectors = np.zeros((steps,len(qubit_indices),3),dtype=np.complex_)
    n = np.zeros(steps, dtype=np.complex_)
    num = qt.num(state.dims[0][cav_index])
    cavity_sim = cav_index is not None
    for i in range(steps):
        dm = qt.ket2dm(state)
        if cavity_sim:
            n[i] = (num* dm.ptrace(cav_index)).tr()
        components = [measure_qubit(dm.ptrace(np.asscalar(i))) for i in qubit_indices]
        bloch_vectors[i] = components
        rho = (rho + do_rk4_step(rho,H,lindblads,tau))
    return bloch_vectors,n
