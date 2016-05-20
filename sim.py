from matplotlib import pyplot as plt
from QDSim import *

# H = 0.5 * (qt.tensor(SZ, I) + qt.tensor(I, SZ)) + qt.tensor(SPlus, SMinus) + qt.tensor(SMinus, SPlus)
L1 = qt.tensor(qt.qeye(5),SMinus,I)
L2 = qt.tensor(qt.qeye(5),I,SMinus)
L3 = qt.tensor(qt.destroy(5),I,I)
H_TC = qt.tensor(qt.num(5),I,I) + qt.tensor(qt.qeye(5),build_S_total(2,SZ)) + qt.tensor(qt.create(5),build_S_total(2,SMinus)) + qt.tensor(qt.destroy(5),build_S_total(2,SPlus))
cav = qt.basis(5,0)
q1 = qt.basis(2,0)
q2 = (qt.basis(2,0)+1j*qt.basis(2,1)).unit()
state = qt.tensor(cav,q1,q2)

# lindblads = [L1,L2]

def test_mc_sim():
    state = qt.tensor(cav,q1,q2)
    return do_jump_mc(state,H_TC,[],1000)

def test_rk4_sim():
    state = qt.tensor(cav,q1,q2)
    return do_rk4(state,H_TC,[],1000)

if __name__ == '__main__':    
    # x1,y1,z1,x2,y2,z2,n = test_mc_sim()
    # b = qt.Bloch()
    # b.add_points([x1,y1,z1])
    # b.add_points([x2,y2,z2])
    # b.show()
    # print(n)
    # plt.hist(np.real(n),10)
    # plt.show()
    # for i in range(x1.size):
    #     b.clear()
    #     b.add_points([x1[:i+1],y1[:i+1],z1[:i+1]])
    #     b.save(dirc='temp')
    import timeit
    print(timeit.Timer("test_mc_sim()",setup="from __main__ import test_mc_sim").repeat(3,1))
    print(timeit.Timer("test_rk4_sim()",setup="from __main__ import test_rk4_sim").repeat(3,1))