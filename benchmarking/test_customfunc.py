from numpy.ctypeslib import ndpointer
from ctypes import *
import numpy as np
import sys


################################
# defining custom types
################################
class PointerWrapper(object):
    """Just like ndpointer, but accept None!"""
    def __init__(self,pointer):
        self.pointer=pointer
    def from_param(self,param):
        if param!=None:
            return self.pointer.from_param(param)
        else:
            return POINTER(c_double).from_param(None)


c_complexdouble = c_double*2

################################

# setting up C function
so_file = "/home/etolley/bluebild/pypeline/src/python-tester.so"
tester_functions = CDLL(so_file)
tester_functions.test_c_int.argtypes = [c_int]
tester_functions.test_c_complex_fromPython.argtypes = [ndpointer(dtype=np.complex128,ndim=1,flags='C')]
tester_functions.test_c_complex_pointer.argtypes = [ndpointer(dtype=np.complex128,ndim=2,flags='C')]
A = 34
C = (np.random.rand(2,2) + 1j*np.random.rand(2,2))
B = np.array([1. + 1.j*2]).astype(np.complex128)
print(C)
tester_functions.test_c_int(A)
tester_functions.test_c_complex_fromPython(B)
tester_functions.test_c_complex_pointer(C)

sys.exit()

'''
# setting up C function
so_file = "/home/etolley/bluebild/pypeline/src/zgemm-splat.so"
custom_functions = CDLL(so_file)



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
'''