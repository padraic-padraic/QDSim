from QDSim import qt
from QDSim.hamiltonian import *
from sympy import sympify

import os
import re
import yaml

#Handy Snippet from StackOverflow to make pyyaml correctly identify floats
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
    '2q_direct': ['w_1', 'w_2', 'g'],
    'full_approx': ['cav_dim', 'w_1', 'w_2', 'w_c', 'g']
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
                  'g': 6e9},
    'full_approx':{'cav_dim':5,
                   'w_1':1e9,
                   'w_2': 1e9,
                   'w_c':5e9,
                   'g': .6e6}
}

H_FUNC = {
    'full': full_hamiltonian,
    '1qb': single_hamiltonian,
    '2q_direct': direct_hamiltonian,
    'full_approx': full_approx
}

class Conf():
    def __init__(self,fname):
        with open(os.path.join(os.path.dirname(__file__),fname), 'r') as f:
            conf = yaml.load(f, Loader=LOADER)
            self.type = conf.get('type', None)
            self.params = conf.get('params', {})
        if self.params:
            self._validate_type()
        self.__H__ = None

    def _validate_type(self):
        """Check that we have all the parameters we need for this type of Hamiltonian"""
        if any([_key not in self.params for _key in H_PARAMS[self.type]]):
            raise KeyError('Missing parameter for this Hamiltonian type!')

    def _process_symb(self, expr):
        """Evaluate a parameter from its symbolic form"""
        expr = sympify(expr)
        _vars = expr.free_symbols
        for var in _vars:
            sub = self.params[var.__str__()]
            if isinstance(sub,str):
                sub = self._process_symb(sub)
            expr = expr.subs(var, sub)
        return float(expr.evalf())
    
    def _load_H(self):
        vals = []
        for _key in H_PARAMS[self.type]:
            val = getattr(self,_key,None)
            if val:
                vals.append(val)
            else:
                print(_key)
                _val = self.params.get(_key, DEFAULTS[self.type][_key])
                if isinstance(_val, str):
                    _val = (self._process_symb(_val))
                vals.append(_val)
                setattr(self,_key,_val)
        self.__H__ = H_FUNC[self.type](*vals)

    @property
    def H(self):
        if not self.__H__:
            self._load_H()
        return self.__H__

    def build_generator_func(params):
        iter_keys = [key for key in params.keys() if isinstance(params[key], list)]

# ----- Methods used to calculate the Swap-Basis transform, taken from 10.1103/PhysRevA.75.032329 ----- #
    @property
    def iSWAP_U(self):
        dim = self.params['cav_dim']
        if self.type != 'full':
            return None
        a = qt.tensor(qt.destroy(dim), I, I)
        sm1 = qt.tensor(qt.qeye(dim), SMinus, I)
        sm2 = qt.tensor(qt.qeye(dim), I, SMinus)
        sp1 = qt.tensor(qt.qeye(dim), SPlus, I)
        sp2 = qt.tensor(qt.qeye(dim), I, SPlus)
        return (
                (self.params['g_1']/self.delta(1)) * (a.dag()*sm1 - a*sp1) +
                (self.params['g_2']/self.delta(2)) * (a.dag()*sm2 - a*sp2)).expm()

    def delta(self, qubit):
        if self.type == '1qb' and qubit > 1:
            raise Exception
        return self.params['w_'+str(qubit)] - self.params['w_c']

    def chi(self, qubit):
        if self.type == '1qb' and qubit > 1:
            raise Exception
        g_factor = self.params['g_'+str(qubit)]
        delta = self.delta(qubit)
        return g_factor*g_factor/delta
