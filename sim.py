from matplotlib import pyplot as plt
from QDSim import *

H = 0.5 * (qt.tensor(SZ, I) + qt.tensor(I, SZ)) + qt.tensor(SPlus, SMinus) + qt.tensor(SMinus, SPlus)
L1 = 0.5 * qt.tensor(SMinus,I)
L2 = 0.5 * qt.tensor(I,SMinus) 
q1 = qt.Qobj(np.array([0,1]))
q2 = qt.Qobj(np.array([0.5,0.5]))
state = qt.tensor(q1,q2)

# lindblads = [L1,L2]

def test_mc_sim():
    state = qt.tensor(q1,q2)
    x1,y1,z1,x2,y2,z2 = do_jump_mc(state,H,[L1,L2],100)

def test_rk4_sim():
    state = qt.tensor(q1,q2)
    x1,y1,z1,x2,y2,z2 = do_rk4(state,H,[L1,L2],100)

if __name__ == '__main__':    
    # x1,y1,z1,x2,y2,z2 = test_mc_sim()
    # b = qt.Bloch()
    # b.add_points([x1,y1,z1])
    # b.add_points([x2,y2,z2])
    # b.show()
    # for i in range(x1.size):
    #     b.clear()
    #     b.add_points([x1[:i+1],y1[:i+1],z1[:i+1]])
    #     b.save(dirc='temp')
    import timeit
    print(timeit.Timer("test_mc_sim()",setup="from __main__ import test_mc_sim").repeat(3,1))
    print(timeit.Timer("test_rk4_sim()",setup="from __main__ import test_rk4_sim").repeat(3,1))