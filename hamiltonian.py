"""Contians functions to build Hamiltonian recipes from a list of Parameters.
If adding a new recipe, you also need to define a 'type' and add to the dictionaries
of parameters, defaults and functions in QDSim.conf_loader.
"""

from QDSim import qt, I, SZ, SPlus, SMinus

def full_hamiltonian(cav_dim, w_1, w_2, w_c, g_1, g_2):
    """Return a QObj denoting the full Hamiltonian including cavity
     and two qubits"""
    return (w_c * qt.tensor(qt.num(cav_dim), I, I) +
            w_1 * qt.tensor(qt.qeye(cav_dim), SZ, I) +
            w_2 * qt.tensor(qt.qeye(cav_dim), I, SZ) +
            g_1 * qt.tensor(qt.create(cav_dim), SMinus, I) +
            g_1 * qt.tensor(qt.destroy(cav_dim), SPlus, I) +
            g_2 * qt.tensor(qt.create(cav_dim), I, SMinus) +
            g_2 * qt.tensor(qt.destroy(cav_dim), I, SPlus))

def single_hamiltonian(cav_dim, w_1, w_c, g_factor):
    """Return a QObj denoting a hamiltonian for one qubit coupled to a
    cavity."""
    return (w_c * qt.tensor(qt.num(cav_dim), I) +
            w_1 * qt.tensor(qt.qeye(cav_dim), I) +
            g_factor * qt.tensor(qt.create(cav_dim), SMinus) +
            g_factor * qt.tensor(qt.destroy(cav_dim), SPlus))

def direct_hamiltonian(w_1, w_2, g_factor):
    """Return a QObj denoting a hamiltonian for two qubits interacting with the
    cavity mode eliminated."""
    return (w_1 * qt.tensor(SZ, I) +
            w_2 * qt.tensor(I, SZ) +
            g_factor * qt.tensor(SPlus, SMinus) +
            g_factor * qt.tensor(SMinus, SPlus))
