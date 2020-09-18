import numpy as np
import numexpr as ne
from numpy import matmul
from scipy.linalg import expm
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
    # I has shape (N_eig, Nheight, N_width)

    return I

# does the same calculations as synthesize but iterating throught
# the matrix stack instead of uisng tensordot. 
# difference between the results is quite large, ~1e-7
def synthesize_stack(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl
    B  = np.zeros((N_antenna, N_height,N_width),dtype=np.float32)
    P  = np.zeros(B.shape,                      dtype=np.complex64)
    PW = np.zeros((N_beam, N_height, N_width),  dtype=np.complex64)
    E  = np.zeros((N_eig, N_height, N_width),   dtype=np.complex64)
    I  = np.zeros(E.shape,                      dtype=np.float32)
    for i in range(N_width): #iterate over N_width
        B[:,:,i]  = matmul(XYZ, pixGrid[:,:,i])
        #P[:,:,i]  = np.exp(a*B[:,:,i])
        ne.evaluate( "exp(A * B)",dict(A=a, B=B[:,:,i]),out=P[:,:,i],casting="same_kind",) 
        PW[:,:,i] = matmul(W.T,P[:,:,i])
        E[:,:,i]  = matmul(V.T, PW[:,:,i])
        I[:,:,i] = E[:,:,i].real ** 2 + E[:,:,i].imag ** 2

    return I

# does the same calculations as synthesize but reshaping
# the matrix stack instead of uisng tensordot. 
def synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)
    B = XYZ @ pixGrid

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    #P = np.exp(a*B) # introduces some differences at fp level
    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

#################################################################################
if __name__ == "__main__":

    data = RandomDataGen()
    synthesize(data.getPixGrid(), *data.getVXYZW(0), data.wl)