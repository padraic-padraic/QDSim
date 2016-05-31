from QDSim.hamiltonian import *
from sympy import sympify

import qutip as qt
import os
import re
import yaml

__all__ = ["Conf"]

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
    'full_approx': ['cav_dim', 'w_1', 'w_2', 'w_c', 'g'],
    'time_dependent': ['cav_dim', 'w_1', 'w_2', 'w_c', 'g_1', 'g_2'],
}

DEFAULTS = {
    'full': {'cav_dim': 5,
             'w_1': 4.9e9,
             'w_2': 4.9e9,
             'w_c': 5e9,
             'g_1': 15e6,
             'g_2': 15e6},
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
                   'g': .6e6},
   'time_dependent': {'cav_dim': 5,
                      'w_1': 6e9,
                      'w_2': 6e9,
                      'w_c': 6e9,
                      'g_1': 5e5,
                      'g_2': 5e5},
}

H_FUNC = {
    'full': full_hamiltonian,
    '1qb': single_hamiltonian,
    '2q_direct': direct_hamiltonian,
    'full_approx': full_approx,
    'time_dependent': interaction_picture
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
