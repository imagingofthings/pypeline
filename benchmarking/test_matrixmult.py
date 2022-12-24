import numpy as np
import numexpr as ne
import timing
import time
from math import exp
from numpy import matmul
from scipy.linalg.blas import dgemm
import matplotlib.pyplot as plt
import sys
from numpy.ctypeslib import ndpointer
from ctypes import *

#numactl --physcpubind=0 python test_matrixmult.py

class PointerWrapper(object):
    """Just like ndpointer, but accept None!"""
    def __init__(self,pointer):
        self.pointer=pointer
    def from_param(self,param):
        if param!=None:
            return self.pointer.from_param(param)
        else:
            return POINTER(c_double).from_param(None)

def getDiff(A,B):
    return  np.average( A - B)

def tensordotCheck():
    ####################################################################
    #### checking how tensordot works on arrays with different dimensions
    # ie A = (550, 3) , B = (3, 248, 124) , out = (550, 248, 124)
    # or A = (12, 24) , B = (24, 248, 124) , out = (12, 248, 124)
    A = np.random.rand(550,3)
    B = np.random.rand(3, 248, 124)
    C = np.tensordot(A, B, axes=1)

    B1 = B[:,:,0]
    C1 = np.tensordot(A, B1, axes=1)

    print( (C[:,:,0].shape))
    print( (C1.shape))

    diff = getDiff(C[:,:,0],C1) #np.average( C[:,:,0] - C1)
    # if the dot product is done on each slice in the 124-sized dim, diff should be zero
    print(diff)

def numexprCheck():

    ####################################################################
    # testing numexpr
    # numexpr does pointwise calculations
    P = np.zeros( (100, 200))
    B = np.random.rand(100,200)
    ne.evaluate(
                "exp(A * B)",
                dict(A=2, B=A),
                out=P,
                casting="same_kind",
            )  # Due to limitations of NumExpr2

    # if numexpr is doing an element-by-element EXP,
    # the two expressions below should evaluate to the same value
    print(P[0,0,0])
    print(exp(2*C[0,0,0]))

    test1 = np.random.rand(10,10)
    test2 = np.random.rand(10,10)
    out = np.zeros( (10,10))
    ne.evaluate("A * B", dict(A=test1, B=test2), out=out, casting="same_kind",)
    print("(A*B)00 = ",out[0,0])
    print("A00 * B00 = ",test1[0,0] * test2[0,0])



def doMMtest(timer, N_iter, N_antenna = 550, N_height = 248, N_width = 124, y= 3):
    timer.reset()
    for i in range(N_iter):
        A = np.random.rand(N_antenna,3).astype(np.float32)
        B = np.random.rand(3, N_height, N_width).astype(np.float32)

        '''timer.start_time("numpy dot")
        C0 = np.zeros( (N_antenna, N_height, N_width),dtype =np.float32)
        for i in range(N_width):
            C0[:,:,i] = np.dot(A, B[:,:,i])
        timer.end_time("numpy dot")
        timer.set_Nops("numpy dot",N_antenna *N_height *N_width * y)'''

        timer.start_time("numpy tensordot")
        C1 = np.tensordot(A, B, axes=1)
        timer.end_time("numpy tensordot")
        timer.set_Nops("numpy tensordot",N_antenna *N_height *N_width * y)

        timer.start_time("numpy matmul")
        C2 = np.zeros( (N_antenna, N_height, N_width),dtype =np.float32)
        for i in range(N_width):
            C2[:,:,i] = matmul(A, B[:,:,i])
        timer.end_time("numpy matmul")
        timer.set_Nops("numpy matmul",N_antenna *N_height *N_width * y)

        '''from ctypes import CDLL
        so_file = "/home/etolley/bluebild/pypeline/custom_matmul/zgemm-splat.so"
        custom_functions = CDLL(so_file)
        timer.start_time("custom matmul")
        C3 = custom_functions.zgemm()

        timer.end_time("custom matmul")'''


        timer.start_time("LAPACK DGEMM")
        C3 = np.zeros( (N_antenna, N_height, N_width))
        for i in range(N_width):
            C3[:,:,i] = dgemm(1., A, B[:,:,i])
        timer.end_time("LAPACK DGEMM")
        timer.set_Nops("LAPACK DGEMM",N_antenna *N_height *N_width * y)
        #print("avg diff wrt tensordot:", getDiff(C1,C0), getDiff(C1,C1), getDiff(C1,C2), getDiff(C1,C3))

    print ("End matrix dimensions: ",N_antenna, N_height, N_width)
    print(timer.summary())

def testCustomFunc():

    so_file = "/home/etolley/bluebild/pypeline/custom_matmul/zgemm-splat.so"
    custom_functions = CDLL(so_file)
    c_complexdouble = c_double*2
    custom_functions.zgemm.argtypes=[c_int, c_int, c_int, c_complexdouble,
                                    PointerWrapper(ndpointer(dtype=np.complex128,ndim=2,flags='C')), c_int,
                                    PointerWrapper(ndpointer(dtype=np.complex128,ndim=2,flags='C')), c_int,
                                    c_complexdouble,
                                    PointerWrapper(ndpointer(dtype=np.complex128,ndim=2,flags='C')), c_int]
    custom_functions.zgemm.restype=None
    
    M = 100
    N = 100
    K = 3
    A = np.random.rand(M,K).astype(np.complex128)
    ldA = M
    B = np.random.rand(K,N).astype(np.complex128)
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128)
    ldC = M
    beta =  c_complexdouble(1,0)
    alpha = c_complexdouble(1,0)

    custom_functions.zgemm(M, N, K, alpha, A, ldA, B, ldB , beta, C, ldC)
    print(C)

if __name__ == "__main__":
    #testCustomFunc()


    # timer
    timer = timing.Timer()
    doMMtest(timer, 1)
    sys.exit()

    (N_antenna, N_height, N_width) = (550, 248, 124)
    (N_antenna, N_height, N_width) = (10000, 10000, 124)
    N_iter = 5
    times = [[], []]
    flops = [[], []]
    names = [[], []]
    sizes = [1,2,3,5, 10, 50, 100, 500, 1000]#,10,50,124]
    #sizes = [2,5,10,50, 100, 1000, 5000, 10000]#,10,50,124]
    for s in sizes:
        doMMtest(timer, N_iter, 550, 248, s)
        for i, t in enumerate( timer.get_times()):
            times[i].append(t)
        for i, f in enumerate( timer.get_Gflops()):
            flops[i].append(f)
        for i, n in enumerate(timer.get_names()):
            names[i] = n

    fig, ax = plt.subplots(ncols=2)
    for i, n in enumerate(names):
        ax[0].plot(sizes, times[i], label=n)
        ax[1].plot(sizes, flops[i], label=n)
    ax[0].legend()
    ax[0].set_xlabel('Size (loop dimension)') #'N (NxN matrix size)'
    ax[0].set_ylabel('Runtime [s]')
    ax[1].legend()
    ax[1].set_xlabel('Size (loop dimension)') #Size (loop dimension)'
    ax[1].set_ylabel('Gflops / s')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.9, hspace=0.25, wspace=0.35)

    fig.show()
    input("Press enter....")
    
    #print(getDiff(C1,C2))
    #print(getDiff(C2,C3))
    #print(getDiff(C3,C1))



