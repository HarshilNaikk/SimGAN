import numpy as np
from scipy import signal
import math
from torch.utils.data import DataLoader, TensorDataset
import torch

Tensor = torch.FloatTensor

class System:
    def __init__(self, zeta=0):
        self.zeta = np.array([[zeta]])
        self.A = np.array([[1, 2],[-2, 1]])
        self.B = np.array([[0.5],[1]])
        # self.C = np.array([[1, 0], [0, 1]])
        # Extra Comment 
        # self.D = np.array([[0, 0], [0, 0]])
    
    def step(self, prev_state, cx=None):
        k = signal.place_poles(self.A,self.B,np.array([0.9,0.8]))
        # prev_state = prev_state.cuda()
        kgmatrix = (k.gain_matrix)
        prev_state = prev_state.cpu()
        u_k = -kgmatrix.dot(prev_state)
        # u_k = u_k.cpu()
        # print("SHAPE OF uk = ")
        # print(u_k)
        
        if cx == None:
            cx = np.array([[0.1*math.sin(prev_state[1])], [0.1*math.cos(prev_state[0])]], dtype='float32').dot(np.array(u_k))*0.25
            cx = cx[np.newaxis, :]
        # Leader Difference Equation: x(k+1) = A*x(k) + B*u_k
        # print("SHAPE OF CX = " + str(np.shape(cx)))
        next_state = Tensor(np.matmul(self.A, prev_state.T, dtype='float32') + np.matmul(self.B, u_k.T, dtype='float32') + np.array(cx, dtype='float32'))
        # print(np.shape(next_state.T))
        error = cx
        # print(np.shape(u_k))
        next_obs = np.concatenate((next_state.T[:, 0], u_k))
        return next_state.T, next_obs, error

    def step_with_predefined_actions(self,prev_state,control_input,cx):
        k = signal.place_poles(self.A,self.B,np.array([0.9,0.8]))
        # prev_state = prev_state.cuda()
        kgmatrix = (k.gain_matrix)
        prev_state = prev_state.cpu()
        u_k = control_input
        # u_k = u_k.cpu()
        print("SHAPE OF uk = " + str((u_k)) )
        u_k = np.reshape(u_k, (1))
        
        if cx == None:
            cx = np.array([[0.1*math.sin(prev_state[1])], [0.1*math.cos(prev_state[0])]], dtype='float32').dot(np.array(u_k))*0.25
            cx = cx[np.newaxis, :]
        # Leader Difference Equation: x(k+1) = A*x(k) + B*u_k
        # print("SHAPE OF CX = " + str(np.shape(cx)))
        next_state = Tensor(np.matmul(self.A, prev_state.T, dtype='float32') + np.matmul(self.B, u_k.T, dtype='float32') + np.array(cx, dtype='float32'))
        # print(np.shape(next_state.T))
        error = cx
        # print(np.shape(u_k))
        next_obs = np.concatenate((next_state.T[:, 0], u_k))
        return next_state.T, next_obs, error

    
    def generate_initial_obs(self):
        x = 2*np.random.uniform(size=2)
        k = signal.place_poles(self.A,self.B,np.array([0.9,0.8]))
        u_k = -k.gain_matrix.dot(x)
        return np.concatenate((x, u_k))