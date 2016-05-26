from QDSim.hamiltonian import *
from sympy import sympify

import re
import yaml

#Handy Snippet from StackOverflow to make yaml correctly identify floats
LOADER = yaml.SafeLoader
LOADER.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

H_PARAMS = {
    'full': ['cav_dim', 'w_1', 'w_2', 'w_c', 'g_1', 'g_2'],
    '1qb': ['cav_dim', 'w_1', 'w_c', 'g'],
    '2q_direct': ['w_1', 'w_2', 'w_c']
}

DEFAULTS = {
    'full': {'cav_dim': 5,
             'w_1': 6e9,
             'w_2': 6e9,
             'w_c': 6e9,
             'g_1': 5e5,
             'g_2': 5e5},
    '1qb': {'cav_dim': 5,
            'w_1': 6e9,
            'w_c':6e9,
            'g': 5e5},
    '2q_direct': {'w_1':6e9,
                  'w_2': 6e9,
                  'g': 6e9}
}

H_FUNC = {
    'full': full_hamiltonian,
    '1qb': single_hamiltonian,
    '2q_direct': direct_hamiltonian
}

def validate_type(_type, params):
    """Check that we have all the parameters we need for this type of Hamiltonian"""
    if any([_key not in params for _key in H_PARAMS[_type]]):
        raise KeyError('Missing parameter for this Hamiltonian type!')

def process_symb(expr, params):
    """Evaluate a parameter from its symbolic form"""
    expr = sympify(expr)
    _vars = expr.free_symbols
    for var in _vars:
        expr.subs(var, params[var.__str__()], eval=False)
    return expr.evalf()

# def 

def load(fname):
    """Load and return the Hamiltonian from a config file."""
    with open(fname, 'r') as f:
        conf = yaml.load(f, Loader=LOADER)
    _type = conf.get('type', None)
    _params = conf.get('params', {})
    if _params: #Otherwise, we fall back to the defaults
        validate_type(_type, _params)
    vals = []
    for _key in H_PARAMS[_type]:
        _val = _params.get(_key, DEFAULTS[_type][_key])
        if isinstance(_val, str):
            vals.append(process_symb(_val, _params))
        else:
            vals.append(_val)
    return H_FUNC[_type](*vals)

