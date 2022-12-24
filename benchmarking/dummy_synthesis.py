import numpy as np
import numexpr as ne
from numpy import matmul
from scipy.linalg import expm
import time
import scipy.constants as constants
import scipy.linalg as linalg

from numpy.ctypeslib import ndpointer
from ctypes import *
from data_gen_utils import RandomDataGen

#local imports
import timing

def make_complexdouble_array(c):
    return np.array([c]).astype(np.complex128)



#################################################################################
# Custom MM
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
    A = A.astype(np.complex128, order = 'F')
    B = B.astype(np.complex128, order = 'F')
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
    A = A.astype(np.complex128, order = 'F')
    B = B.astype(np.complex128, order = 'F')
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M
    alpha = make_complexdouble_array(a)


    # function call
    custom_functions.zgemmexp(M, N, K, alpha, A, ldA, B, ldB, C, ldC)

    return C

def dgemm(A,B, a = 1, b = 0):
    t0 = time.process_time() 
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/dgemm-simple.so"
    custom_functions = CDLL(so_file)
    custom_functions.dgemm.argtypes=[c_int, c_int, c_int,
                                    c_double, #alpha
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    c_double, # beta
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    (M,K) = A.shape
    (K,N) = B.shape
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.float64, order='F')
    ldC = M

    # function call
    t1 = time.process_time() 
    custom_functions.dgemm(M, N, K, a, A, ldA, B, ldB, b, C, ldC)

    return C

def dgemmexp(A,B, a = 1):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/dgemm-simple.so"
    custom_functions = CDLL(so_file)
    custom_functions.dgemmexp.argtypes=[c_int, c_int, c_int,
                                    c_double, #alpha
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
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

    # function call
    custom_functions.dgemmexp(M, N, K, a, A, ldA, B, ldB, C, ldC)

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
    t0 = time.process_time()
    B = XYZ @ pixGrid

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 
    #P = np.exp(a*B) # introduces some differences at fp level
    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def zgemm_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    #V = np.asfortranarray(V)
    #W = np.asfortranarray(W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    t1 = time.process_time()
    B = zgemm(XYZ,pixGrid)

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def dgemm_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    #V = np.asfortranarray(V)
    #W = np.asfortranarray(W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    a = 1j * 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    t1 = time.process_time()
    B = dgemm(XYZ,pixGrid)

    P = np.zeros(B.shape,dtype=np.complex64)
    ne.evaluate( "exp(A * B)",dict(A=a, B=B),out=P,casting="same_kind",) 

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def zgemmexp_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    #V = np.asfortranarray(V) 
    #W = np.asfortranarray(W)

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    # dgemmexp will apply the imaginary component to a
    a = 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    P = zgemmexp(XYZ,pixGrid,a)

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

def dgemmexp_synthesize_reshape(pixGrid, V, XYZ, W, wl):  
    N_antenna, N_beam = W.shape
    N_height, N_width = pixGrid.shape[1:] 
    N_eig = V.shape[1]

    pixGrid = pixGrid / linalg.norm(pixGrid, axis=0)
    XYZ = XYZ - XYZ.mean(axis=0)

    # dgemmexp will apply the imaginary component to a
    a = 2 * np.pi / wl

    #flatten the height and width
    pixGrid = pixGrid.reshape(pixGrid.shape[0], N_height * N_width)

    P = dgemmexp(XYZ,pixGrid,a)

    PW = W.T @ P
    E  = V.T @ PW
    I  = E.real ** 2 + E.imag ** 2
    I  = I.reshape(I.shape[0],N_height, N_width)

    return I

#################################################################################

def test_dgemm(data, string = ""):
    A2 = data.getXYZ()
    B2 = data.getPixGrid()
    B2 = B2 / linalg.norm(B2, axis=0)
    A2 = A2 - A2.mean(axis=0)
    B2 = B2.reshape(B2.shape[0], B2.shape[1] * B2.shape[2])

    t2 = time.process_time()
    result = dgemm(A2,B2)
    print("DGEMM", string, "time", time.process_time() - t2)



if __name__ == "__main__":

    # parameters
    frequency = 145e6
    wl = constants.speed_of_light / frequency

    data = RandomDataGen(64)
    timer = timing.Timer()

    pix = data.getPixGrid()

    for t in range(0,1):
        (V, XYZ, W) = data.getVXYZW(t)


        # call the dummy synthesis kernal
        timer.start_time("Dummy synthesis")
        stat_dum  = synthesize(pix,V,XYZ,W, wl)
        timer.end_time("Dummy synthesis")

        # call an alternate dummy synthesis kernel which reshapes the matrices
        timer.start_time("Reshaped dummy synthesis")
        stat_sdum = synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("Reshaped dummy synthesis")

        # call an alternate dummy synthesis kernel which uses a special DGEMM
        timer.start_time("DGEMM dummy synthesis")
        stat_gdum = dgemm_synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("DGEMM dummy synthesis")

        # call an alternate dummy synthesis kernel which uses a special ZGEMM
        #timer.start_time("ZGEMM dummy synthesis")
        #stat_zdum = zgemm_synthesize_reshape(pix,V,XYZ,W, wl)
        #timer.end_time("ZGEMM dummy synthesis")

        # call an alternate dummy synthesis kernel which uses an extra special ZGEMM
        timer.start_time("DGEMMexp dummy synthesis")
        stat_gexpdum =  dgemmexp_synthesize_reshape(pix,V,XYZ,W, wl)
        timer.end_time("DGEMMexp dummy synthesis")
        print("Avg diff between dummy & dummy reshape synthesizers:", np.average( stat_dum - stat_sdum))
        print("Avg diff between dummy & DGEMM synthesizers:", np.max( np.abs(stat_dum - stat_gdum)))
        print("Avg diff between dummy & DGEMMexp synthesizers:", np.max( np.abs(stat_dum - stat_gexpdum)))
        print("Avg diff between DGEMM & DGEMMexp synthesizers:", np.max( np.abs(stat_gdum - stat_gexpdum)))

    print(timer.summary())




