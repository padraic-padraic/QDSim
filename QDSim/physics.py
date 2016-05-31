"""Helpful states and operators"""
import numpy as np
import qutip as qt

__all__ = ["I", "SX", "SY", "SZ", "iSWAP", "root_iSWAP", "SPlus", "SMinus"]

I = qt.qeye(2)
SX = qt.sigmax()
SY = qt.sigmay()
SZ = qt.sigmaz()
iSWAP = np.zeros((4,4), dtype=np.complex_)
iSWAP[0, 0], iSWAP[3,3] = 1, 1 
iSWAP[1,2],iSWAP[2,1] = 1j,1j
iSWAP = qt.Qobj(iSWAP,dims=[[2,2],[2,2]])
root_iSWAP = iSWAP.sqrtm()
#We're treating '1' as the excited state
SPlus = qt.create(2)
SMinus = qt.destroy(2)