"""Contains functions to build Hamiltonian recipes from a list of Parameters.
If adding a new recipe, you also need to define a 'type' and add to the dictionaries
of parameters, defaults and functions in QDSim.conf_loader.
"""

from QDSim.physics import I, SZ, SPlus, SMinus

import qutip as qt

hbar = 1.05e-34
__all__ = []

def full_hamiltonian(cav_dim, w_1, w_2, w_c, g_1, g_2):
    """Return a QObj denoting the full Hamiltonian including cavity
     and two qubits"""
    a = qt.destroy(cav_dim)
    num = a.dag() * a
    return (
            g_1 * qt.tensor(qt.create(cav_dim), SMinus, I) +
            g_1 * qt.tensor(qt.destroy(cav_dim), SPlus, I) +
            g_2 * qt.tensor(qt.create(cav_dim), I, SMinus) +
            g_2 * qt.tensor(qt.destroy(cav_dim), I, SPlus) +
            w_c * qt.tensor(num, I, I) +
            0.5 * w_1 * qt.tensor(qt.qeye(cav_dim), SZ, I) +
            0.5 * w_2 * qt.tensor(qt.qeye(cav_dim), I, SZ))

def single_hamiltonian(cav_dim, w_1, w_c, g_factor):
    """Return a QObj denoting a hamiltonian for one qubit coupled to a
    cavity."""
    return (w_c * qt.tensor(qt.num(cav_dim), I) +
            0.5 * w_1 * qt.tensor(qt.qeye(cav_dim), I) +
            g_factor * qt.tensor(qt.create(cav_dim), SMinus) +
            g_factor * qt.tensor(qt.destroy(cav_dim), SPlus))

def full_approx(cav_dim, w_1, w_2, w_c, g_factor):
    a = qt.destroy(cav_dim)
    num = a.dag() * a
    return (
            g_factor * qt.tensor(qt.qeye(cav_dim), SMinus, SPlus) +
            g_factor * qt.tensor(qt.qeye(cav_dim), SPlus, SMinus) +
            w_c * qt.tensor(num, I, I) +
            0.5 * w_1 * qt.tensor(qt.qeye(cav_dim), SZ, I) +
            0.5 * w_2 * qt.tensor(qt.qeye(cav_dim), I, SZ))

def direct_hamiltonian(w_1, w_2, g_factor):
    """Return a QObj denoting a hamiltonian for two qubits interacting with the
    cavity mode eliminated."""
    return (
            0.5 * w_1 * qt.tensor(SZ, I) +
            0.5 * w_2 * qt.tensor(I, SZ) +
            g_factor * qt.tensor(SPlus, SMinus) +
            g_factor * qt.tensor(SMinus, SPlus))

def interaction_picture(cav_dim, w_1, w_2, w_c, g_1, g_2):
    delta_1 = abs(w_1 - w_c)
    delta_2 = abs(w_2 - w_c)
    a = qt.destroy(cav_dim)
    H0 = qt.tensor(qt.qeye(cav_dim), I, I)
    H1 = [g_1 * qt.tensor(a.dag(),SMinus,I), 'exp(-1j *'+str(delta_1)+'*t)']
    H2 = [g_2 * qt.tensor(a.dag(),I,SMinus), 'exp(-1j *'+str(delta_2)+'*t)']
    H3 = [g_1 * qt.tensor(a,SPlus,I), 'exp(1j *'+str(delta_1)+'*t)']
    H4 = [g_2 * qt.tensor(a,I,SPlus), 'exp(1j *'+str(delta_2)+'*t)']
    return [H0,H1,H2,H3,H4]

