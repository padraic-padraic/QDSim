import qutip as qt
import numpy as np

from random import random
#IDK why qutip doesn't just make these objects?
I = qt.qeye(2)
SX = qt.sigmax()
SY = qt.sigmay()
SZ = qt.sigmaz()
SPlus = qt.sigmap()
SMinus = qt.sigmam()
# Do a lindblad-ish thing
# Calculate d-pho/dt for each step totally w/ fine-grained t
# Alternatively, use the MC recipe from harroche


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


def get_prob(state,L):
    """Get the probability for each jump operator"""
    return (L.dag()*L*qt.ket2dm(state)).tr()

def get_bloch_tuple(dm):
    u = (SX*dm).tr()
    v = (SY*dm).tr()
    w = (SZ*dm).tr()
    return u,v,w


def get_M0(H,lindblads,tau):
    J = qt.Qobj(np.zeros_like(H.data.todense()),dims=[[2,2],[2,2]])
    for Op in lindblads:
        J += Op.dag()*Op
    J *= 0.5
    return qt.tensor(I,I) - tau*(1j*H - J)

def do_jump_mc(state,H,lindblads,steps=1000,tau=1/100):
    """Quantum Jump Monte Carlo for simulating Master Equation dynamics. 
        Implemented following the recipe in Harroche & Raimond 
    """
    #Prebuild arrays
    tau = 1/steps
    xp_1 = np.zeros(steps, dtype=np.complex_)
    yp_1 = np.zeros_like(xp_1, dtype=np.complex_)
    zp_1 = np.zeros_like(xp_1, dtype=np.complex_)
    xp_2 = np.zeros_like(xp_1, dtype=np.complex_)
    yp_2 = np.zeros_like(xp_1, dtype=np.complex_)
    zp_2 = np.zeros_like(xp_1, dtype=np.complex_)
    #Check if our lindblads are time/state dependent
    time_dependent = any([hasattr(Op,'__call__')] for Op in lindblads)
    #Build the regular old evolution operator M0
    M0 = get_M0(H, lindblads,tau)
    for i in range(steps):
        #Refresh lindblads if time dependent
        q1 = qt.ket2dm(state).ptrace(0)
        q2 = qt.ket2dm(state).ptrace(1)
        x1,y1,z1 = get_bloch_tuple(q1)
        x2,y2,z2 = get_bloch_tuple(q2)
        xp_1[i] = x1
        yp_1[i] = y1
        zp_1[i] = z1
        xp_2[i] = x2
        yp_2[i] = y2
        zp_2[i] = z2
        if time_dependent:
            #Refresh lindlabds if necessary
            lindblads = [Op(state, i*tau) if type(Op)=='function' else Op for Op in lindblads]
            #Refresh J, 'regular' evolution operator
            M0 = get_M0(H,lindblads,tau)
        p_jump = np.array([tau*get_prob(state, Op) for Op in lindblads], dtype=np.complex_)
        p_nojump = 1.-np.sum(p_jump)
        r = random()
        if r > p_nojump:
            print('Jump on round ' + str(i))
            #Find the correct jump operator
            n = find_jump(r,p_jump)
            if n is not None:
                state = lindblads[n]*state / np.sqrt(p_jump[n]/tau)
            else:#Fallback option, sometimes necessary due to floating point weirdness
                state = M0*state / np.sqrt(p_nojump)
        else:
            state = M0*state / np.sqrt(p_nojump)
        state = state.unit()    
    return xp_1,yp_1,zp_1,xp_2,yp_2,zp_2

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
    xp_1 = np.zeros(steps, dtype=np.complex_)
    yp_1 = np.zeros_like(xp_1, dtype=np.complex_)
    zp_1 = np.zeros_like(xp_1, dtype=np.complex_)
    xp_2 = np.zeros_like(xp_1, dtype=np.complex_)
    yp_2 = np.zeros_like(xp_1, dtype=np.complex_)
    zp_2 = np.zeros_like(xp_1, dtype=np.complex_)
    for i in range(steps):
        q1 = rho.ptrace(0)
        q2 = rho.ptrace(1)
        x1,y1,z1 = get_bloch_tuple(q1)
        x2,y2,z2 = get_bloch_tuple(q2)
        xp_1[i] = x1
        yp_1[i] = y1
        zp_1[i] = z1
        xp_2[i] = x2
        yp_2[i] = y2
        zp_2[i] = z2
        rho = (rho + do_rk4_step(rho,H,lindblads,tau))
    return xp_1,yp_1,zp_1,xp_2,yp_2,zp_2
