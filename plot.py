"""Useful routines for making graphs from the output of a given simulation"""
from matplotlib import pyplot as plt
from QDSim import np, qt

def load_results(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [l.strip for l in lines]
    lines = [l.split("\t") for l in lines]
    for i, _l in enumerate(lines):
        lines[i] = map(float, _l)
    return np.array(lines, dtype=np.float_)
