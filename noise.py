from QDSim import np, parse_dims, qt, SMinus

K_B = 1.38e-23
PLANCK = 6.63e-34

def get_n_therm(freq,T):
    return (K_B * T) / (PLANCK * freq)

def thermal_state(freq, T, cav_dim=5):
    n_therm = get_n_therm(freq, T)
    return qt.Qobj(np.sqrt((qt.num(cav_dim)
                  *qt.thermal_dm(cav_dim, n_therm)).diag()))

def cavity_loss(kappa, freq, T, dims):
    qubit_indices,cav_index = parse_dims(dims)
    cav_dim = dims[cav_index]
    n_therm = get_n_therm(freq, T)
    ops = [0]*len(dims)
    ops[cav_index] = qt.destroy(cav_dim)
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(kappa * (1+n_therm) / 2) * qt.tensor(ops)

def thermal_in(kappa, freq, T, dims):
    qubit_indices,cav_index = parse_dims(dims)
    cav_dim = dims[cav_index]
    n_therm = get_n_therm(freq, T)
    ops = [0]*len(dims)
    ops[cav_index] = qt.create(cav_dim)
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(kappa * n_therm / 2) * qt.tensor(ops)

def relaxation(rate, dims, qubit=1):
    qubit_indices,cav_index = parse_dims(dims)
    if cav_index:
        cav_dim = dims[cav_index]
        ops = [0]*len(dims)
        ops[cav_index] = qt.qeye(cav_dim)
    else:
        ops = [0,0]
        ops[qubit-1] = SMinus
    ops = [qt.qeye(2) if op == 0 else op for op in ops]
    return np.sqrt(rate) * qt.tensor(ops)
