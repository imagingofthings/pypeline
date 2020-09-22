from numpy.ctypeslib import ndpointer
from ctypes import *
import numpy as np

class PointerWrapper(object):
    """Just like ndpointer, but accept None!"""
    def __init__(self,pointer):
        self.pointer=pointer
    def from_param(self,param):
        if param!=None:
            return self.pointer.from_param(param)
        else:
            return POINTER(c_double).from_param(None)


# setting up C function
so_file = "/home/etolley/bluebild/pypeline/custom_matmul/zgemm-splat.so"
custom_functions = CDLL(so_file)

c_complexdouble = c_double*2

custom_functions.zgemm.argtypes=[c_int, c_int, c_int, c_complexdouble,
                                PointerWrapper(ndpointer(dtype=np.complex128,ndim=2,flags='C')), c_int,
                                PointerWrapper(ndpointer(dtype=np.complex128,ndim=2,flags='C')), c_int,
                                c_complexdouble,
                                PointerWrapper(ndpointer(dtype=np.complex128,ndim=2,flags='C')), c_int]

custom_functions.zgemm.restype=None

# setting up inputs
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

# function call
custom_functions.zgemm(M, N, K, alpha, A, ldA, B, ldB , beta, C, ldC)

print(C)