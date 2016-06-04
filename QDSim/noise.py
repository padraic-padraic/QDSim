import numpy as np
import qutip as qt

from QDSim.physics import SZ, SMinus
from QDSim.solvers import parse_dims

__all__ =["get_n_therm", "thermal_state", "cavity_loss", "thermal_in",
          "decoherence", "relaxation", "get_lind_list"]

K_B = 1.38e-23
PLANCK = 6.63e-34

def get_n_therm(freq,T):
    return (K_B * T) / (PLANCK * freq)

def thermal_state(freq, T, cav_dim=5):
    n_therm = get_n_therm(freq, T)
    return qt.coherent(5,n_therm)

def cavity_loss(Q, freq, T, dims):
    kappa = Q/freq
    qubit_indices,cav_index = parse_dims(dims)
    cav_dim = dims[cav_index]
    n_therm = get_n_therm(freq, T)
    ops = [0]*len(dims)
    ops[cav_index] = qt.destroy(cav_dim)
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(kappa * (1+n_therm) / 2) * qt.tensor(ops)

def thermal_in(Q, freq, T, dims):
    kappa = Q/freq
    qubit_indices,cav_index = parse_dims(dims)
    cav_dim = dims[cav_index]
    n_therm = get_n_therm(freq, T)
    ops = [0]*len(dims)
    ops[cav_index] = qt.create(cav_dim)
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(kappa * n_therm / 2) * qt.tensor(ops)

def decoherence(rate, dims, qubit=1):
    qubit_indices,cav_index = parse_dims(dims)
    if cav_index is not None:
        cav_dim = dims[cav_index]
        ops = [0]*len(dims)
        ops[cav_index] = qt.qeye(cav_dim)
        ops[qubit_indices[qubit-1]] = SZ
    else:
        ops = [0,0]
        ops[qubit-1] = SMinus
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(rate) * qt.tensor(ops)

def relaxation(rate, dims, qubit=1):
    qubit_indices,cav_index = parse_dims(dims)
    if cav_index is not None:
        cav_dim = dims[cav_index]
        ops = [0]*len(dims)
        ops[cav_index] = qt.qeye(cav_dim)
        ops[qubit_indices[qubit-1]] = SMinus
    else:
        ops = [0,0]
        ops[qubit-1] = SMinus
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(rate) * qt.tensor(ops)

def get_lind_list(w_C, Qfactor, T, gamma_1, gamma_2, dims):
    cav_loss = cavity_loss(Qfactor, w_C, T, dims)
    therm_in = thermal_in(Qfactor, w_C, T, dims)
    d1 = decoherence(gamma_2, dims, 1)
    d2 = decoherence(gamma_2, dims, 2)
    r1 = decoherence(gamma_1, dims, 1)
    r2 = decoherence(gamma_1, dims, 2)
    return [cav_loss, therm_in, d1, d2, r1, r2]
