#python
import time
import numpy as np
np.set_printoptions(precision=3)

th = {}
pi = np.pi

def DH_table(zeta):
    DH = np.zeros((6, 4))
    DH[0,:] = [0.06346, d2r(0),     0.495,      zeta[0]]
    DH[1,:] = [0.175,   d2r(-90),   0,          zeta[1]-(pi/2)]
    DH[2,:] = [1.095,   d2r(0),     0,          zeta[2]]
    DH[3,:] = [0.175,   d2r(-90),   1.27,       zeta[3]]
    DH[4,:] = [0,       d2r(90),    0,          zeta[4]]
    DH[5,:] = [0,       d2r(-90),   0.26494,    zeta[5]]
    return DH
    
def d2r(val):
    return val*pi/180
    
def r2d(val):
    return val*180/pi
    
def Homogeneous(DH, one, two):
    H = np.eye(4)
    for i in range(one, two):
        Tx = Tranx(DH[i][0])
        Rx = Rotx(DH[i][1])
        Tz = Tranz(DH[i][2])
        Rz = Rotz(DH[i][3])
        H = H @ Tx @ Rx @ Tz @ Rz
    return H
    
def T_pos_euler(matrix):
    position = matrix[:3, 3]
    r = matrix[:3, :3]
    psi = np.arctan2(-r[1][2],r[2][2])
    phi = np.arcsin(r[0][2])
    theta = np.arctan2(-r[0][1],r[0][0])
    euler = np.array([(psi), (phi), (theta)])
    return position, euler
        
def Tranx(val):
    out = np.zeros((4, 4))
    out[0,:] = [1, 0, 0, val]
    out[1,:] = [0, 1, 0, 0]
    out[2,:] = [0, 0, 1, 0]
    out[3,:] = [0, 0, 0, 1]
    return out

def Tranz(val):
    out = np.zeros((4, 4))
    out[0,:] = [1, 0, 0, 0]
    out[1,:] = [0, 1, 0, 0]
    out[2,:] = [0, 0, 1, val]
    out[3,:] = [0, 0, 0, 1]
    return out

def Rotx(val):
    out = np.zeros((4, 4))
    out[0,:] = [1, 0, 0, 0]
    out[1,:] = [0, np.cos(val), -np.sin(val), 0]
    out[2,:] = [0, np.sin(val), np.cos(val), 0]
    out[3,:] = [0, 0, 0, 1]
    return out    
    
def Rotz(val):
    out = np.zeros((4, 4))
    out[0,:] = [np.cos(val), -np.sin(val), 0, 0]
    out[1,:] = [np.sin(val), np.cos(val), 0, 0]
    out[2,:] = [0, 0, 1, 0]
    out[3,:] = [0, 0, 0, 1]
    return out   

def find_J(DH):
    r_6E_6 = np.append(T_pos_euler(Homogeneous(DH, 0, 6))[0],1)
    J = np.zeros((6, len(DH)))
    for i in range(len(DH)):
        dum = Homogeneous(DH, i+1,6) @ r_6E_6
        r_iE_6 = dum[:-1]
        T_i_0 = Homogeneous(DH, 0, i+1)
        R_i_0 = T_i_0[:3, :3]
        r_iE_0 = R_i_0 @ r_iE_6
        k_i_0 = R_i_0[:, 2]
        J[0:3, i] = np.cross(k_i_0,r_iE_0)
        J[3:6, i] = k_i_0
    return J
    
def inverse_kinematics(theta, Xd, max_iterations=10, tolerance=1e-4, lr=0.5):
    for i in range(max_iterations):
        DH_ = DH_table(theta)
        J = find_J(DH_)
        J_pseudo = np.linalg.pinv(J)
        b = T_pos_euler(Homogeneous(DH_, 0, 6))
        a = np.hstack((b[0], b[1]))
        error = (Xd[:3] - a[:3])
        del_theta = lr * J_pseudo @ np.pad(error, (0, 3), 'constant')
        theta += del_theta
        if np.mean(error) < tolerance:
            return theta
    return theta
    
def cubic_polynomial(hdl_j, u0, uf, tf, theta, end = False, du0 = 0, duf = 0):
    print(f"go to {uf}")
    a0 = u0
    a1 = du0
    a2 = (3/(tf**2)) * (uf - u0) - ((3/tf) * du0) - ((1/tf) * duf)
    a3 = ((-2/(tf**3)) * (uf - u0)) + ((1/(tf**2)) * (duf + 2*du0))
    start_t = time.time()
    while True:
        if (time.time()-start_t > tf):
            if end == False:
                break
            for count, hdl in enumerate(hdl_j):
                sim.setJointTargetPosition(hdl, theta_bc[count])
        else:
            t = time.time() - start_t
            ut = a0 + a1*t + a2*(t**2) + a3*(t**3)
            theta = inverse_kinematics(theta, ut)
            for count, hdl in enumerate(hdl_j):
                sim.setJointTargetPosition(hdl, theta[count])
            theta_bc = theta
            
        sim.switchThread()
    return theta
    
def path_planning(hdl_j, positions, speeds, loop = False):
    theta = np.zeros(6)
    stat = 1
    while True:
        for count in range(len(positions)-1):
            if loop:
                theta = cubic_polynomial(hdl_j,positions[count],positions[count+1], speeds[count], theta)
            else:
                if len(positions)-2 == count:
                    theta = cubic_polynomial(hdl_j,positions[count],positions[count+1], speeds[count], theta, True)
                else:
                    theta = cubic_polynomial(hdl_j,positions[count],positions[count+1], speeds[count], theta)
    
def sysCall_thread():
    hdl_j = []
    hdl_j.append(sim.getObject("/IRB4600/joint"))
    hdl_j.append(sim.getObject("/IRB4600/joint/link/joint"))
    hdl_j.append(sim.getObject("/IRB4600/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/IRB4600/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/IRB4600/joint/link/joint/link/joint/link/joint/link/joint"))
    hdl_j.append(sim.getObject("/IRB4600/joint/link/joint/link/joint/link/joint/link/joint/link/joint"))
    hdl_end = sim.getObject("/IRB4600/EndPoint")
    
    speeds = [5,10,5,10]
     
    positions = [[1.7734, 0, 1.765],
                [1.90303, 0.1951, 1.08533],
                [0.78901, 0.22286, 1.89798],
                [0.6, 0.6, 1],
                [1.7734, 0, 1.765]]
                
    th =[.1,.2,.3,.4,.5,.6]

    path_planning(hdl_j, np.array(positions), speeds, True)
