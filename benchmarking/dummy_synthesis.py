import numpy as np
import numexpr as ne
from numpy import matmul
from scipy.linalg import expm
import time
import scipy.constants as constants
import scipy.linalg as linalg

from numpy.ctypeslib import ndpointer
from ctypes import *

#local imports
import timing

def make_complexdouble_array(c):
    return np.array([c]).astype(np.complex128)

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
        #return pixGrid.astype(self.ftype)
        return pixGrid.astype(self.ctype)

    # visibilities matrix, (N_beam, N_eig) complex-valued eigenvectors.
    def getV(self, i):
        V = np.random.rand(self.N_beam, self.N_eig)-0.5 + 1j*np.random.rand(self.N_beam, self.N_eig)-0.5j
        return V.astype(self.ctype)

    #  (N_antenna, 3) Cartesian instrument geometry.
    def getXYZ(self, i):
        XYZ = np.random.rand(self.N_antenna,3)
        #return XYZ.astype(self.ftype)
        return XYZ.astype(self.ctype)

    # beamforming weights (N_antenna, N_beam) synthesis beamweights.
    def getW(self, i):
        W = np.random.rand(self.N_antenna, self.N_beam)-0.5 + 1j*np.random.rand(self.N_antenna, self.N_beam)-0.5j
        return W.astype(self.ctype)

    def getVXYZW(self, i):
        return (self.getV(i),self.getXYZ(i),self.getW(i))

#################################################################################
# Custom zgemm
#################################################################################
def zgemm(A,B, a = 1, b = 0):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/zgemm-splat.so"
    custom_functions = CDLL(so_file)
    custom_functions.zgemm.argtypes=[c_int, c_int, c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), #alpha
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), # beta
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M
    alpha = make_complexdouble_array(a)
    beta =  make_complexdouble_array(b)


    # function call
    custom_functions.zgemm(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC)

    return C

def zgemmexp(A,B, a , b = 0):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/zgemm-splat.so"
    custom_functions = CDLL(so_file)
    custom_functions.zgemmexp.argtypes=[c_int, c_int, c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), #alpha
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M
    alpha = make_complexdouble_array(a)


    # function call
    custom_functions.zgemmexp(M, N, K, alpha, A, ldA, B, ldB, C, ldC)

    return C

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

def custom_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)
    B = zgemm(XYZ,pixGrid)

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    #P = np.exp(a*B) # introduces some differences at fp level
    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def customexp_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)
    P = zgemmexp(XYZ,pixGrid,a)

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

#################################################################################
if __name__ == "__main__":

    # parameters
    frequency = 145e6
    wl = constants.speed_of_light / frequency

    data = RandomDataGen(64)
    pix = data.getPixGrid()
    timer = timing.Timer()

    for t in range(1,20):
        (V, XYZ, W) = data.getVXYZW(t)



        # call an alternate dummy synthesis kernel which reshapes the matrices
        timer.start_time("Reshaped dummy synthesis")
        stat_sdum = synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("Reshaped dummy synthesis")

        # call the dummy synthesis kernal
        timer.start_time("Dummy synthesis")
        stat_dum  = synthesize(pix,V,XYZ,W, wl)
        timer.end_time("Dummy synthesis")

        # call an alternate dummy synthesis kernel which uses a special ZGEMM
        timer.start_time("ZGEMM dummy synthesis")
        stat_zdum = custom_synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("ZGEMM dummy synthesis")

        # call an alternate dummy synthesis kernel which uses an extra special ZGEMM
        timer.start_time("ZGEMMexp dummy synthesis")
        stat_zexpdum =  customexp_synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("ZGEMMexp dummy synthesis")
        print("Avg diff between dummy & dummy reshape synthesizers:", np.average( stat_dum - stat_sdum))
        print("Avg diff between dummy & ZGEMM synthesizers:", np.max( np.abs(stat_dum - stat_zdum)))
        print("Avg diff between dummy & ZGEMMexp synthesizers:", np.max( np.abs(stat_dum - stat_zexpdum)))
        print("Avg diff between ZGEMM & ZGEMMexp synthesizers:", np.max( np.abs(stat_zdum - stat_zexpdum)))

    print(timer.summary())
    #synthesize(data.getPixGrid(), *data.getVXYZW(0), data.wl)

