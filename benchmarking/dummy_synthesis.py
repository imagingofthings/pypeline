import numpy as np
import numexpr as ne
from numpy import matmul
import time
import scipy.constants as constants
import scipy.linalg as linalg

#################################################################################
# Data Generator
#################################################################################
class RandomDataGen():
    def __init__(self, precision = 32):

        if (precision == 32):
            self.ftype = np.float32
            self.ctype = np.complex64
        elif (precision == 64):
            self.ftype = np.float64
            self.ctype = np.complex128
        else:
            raise Exception("Precision {0} not known".format(precision))

        # input parameters
        self.N_height  = 248
        self.N_width   = 124
        self.N_antenna = 550
        self.N_beam = 24
        self.N_eig  = 12
        frequency = 145e6
        self.wl = constants.speed_of_light / frequency

    # pixel grid must have dimensions (3, N_height, N_width).
    def getPixGrid(self):
        pixGrid = np.random.rand(3, self.N_height, self.N_width)*2-1
        return pixGrid.astype(self.ftype)

    # visibilities matrix, (N_beam, N_eig) complex-valued eigenvectors.
    def getV(self, i):
        V = np.random.rand(self.N_beam, self.N_eig)-0.5 + 1j*np.random.rand(self.N_beam, self.N_eig)-0.5j
        return V.astype(self.ctype)

    #  (N_antenna, 3) Cartesian instrument geometry.
    def getXYZ(self, i):
        XYZ = np.random.rand(self.N_antenna,3)
        return XYZ.astype(self.ftype)

    # beamforming weights (N_antenna, N_beam) synthesis beamweights.
    def getW(self, i):
        W = np.random.rand(self.N_antenna, self.N_beam)-0.5 + 1j*np.random.rand(self.N_antenna, self.N_beam)-0.5j
        return W.astype(self.ctype)

    def getVXYZW(self, i):
        return (self.getV(i),self.getXYZ(i),self.getW(i))

#################################################################################
# Synthesis kernel
#################################################################################
def synthesize(pixGrid, V, XYZ, W, wl):    

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl
    B = np.tensordot(XYZ, pixGrid, axes=1)
    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    # P has shape (N_antenna,  N_height, N_width)

    PW = np.tensordot(W.T, P, axes=1)
    # PW has shape (N_beam, N_height, N_width)

    E = np.tensordot(V.T, PW, axes=1)
    I = E.real ** 2 + E.imag ** 2
    print(I.shape)
    return I
'''
def synthesize_stack(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid[1:]  

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl
    #B = np.zeros((N_antenna, N_height,N_width),dtype=np.complex64)
    B = np.tensordot(XYZ, pixGrid, axes=1)
    P = np.zeros(B.shape,dtype=np.complex64)
    for i in range(N_width): #iterate over N_width
    	#B[:,:,i] = matmul(XYZ, pixGrid[:,:,i])
    	P[:,:,i] = np.exp(a*B[:,:,i])
    #ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    # P has shape (N_antenna,  N_height, N_width)

    PW = np.tensordot(W.T, P, axes=1)
    # PW has shape (N_beam, N_height, N_width)

    E = np.tensordot(V.T, PW, axes=1)
    I = E.real ** 2 + E.imag ** 2
    print(I.shape)
    return I
'''

#################################################################################
if __name__ == "__main__":

    data = RandomDataGen()
    synthesize(data.getPixGrid(), *data.getVXYZW(0), data.wl)