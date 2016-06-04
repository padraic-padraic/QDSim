"""Useful routines for making graphs from the output of a given simulation"""
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import qutip as qt

__all__ = ["load_results"]

def load_results(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.split("\t") for l in lines]
    for i, _l in enumerate(lines):
        lines[i] = list(map(float, _l))
    return np.array(lines, dtype=np.float_)

def animate_bloch(outname, vectors):
    b = qt.bloch()
    for i, vec in enumerate(vectors):
        b.add_states(b[0], b[1], b[2])
        b.render()
        b.savefig(outname+'_i.png')
    
