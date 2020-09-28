from numpy.ctypeslib import ndpointer
from ctypes import *
import numpy as np
import numexpr as ne
import sys
import faulthandler
import time as time

import os
os.environ['NUMEXPR_NUM_THREADS'] = '1'

#numactl --physcpubind=0 python test_customfunc.py

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


def make_complexdouble_array(c):
    return np.array([c]).astype(np.complex128)

################################

'''
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

def call_dgemm(M,N,K,A,B, a = 1, b = 0):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/dgemm-simple.so"
    custom_functions = CDLL(so_file)
    custom_functions.dgemm.argtypes=[c_int, c_int, c_int,
                                    c_double, #alpha
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    c_double, # beta
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int]
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.float64, order='F')
    ldC = M

    # function call
    faulthandler.enable()
    custom_functions.dgemm(M, N, K, a, A, ldA, B, ldB, b, C, ldC)

    return C

def call_dgemmexp(M,N,K,A,B, a = 1):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/dgemm-simple.so"
    custom_functions = CDLL(so_file)
    custom_functions.dgemmexp.argtypes=[c_int, c_int, c_int,
                                    c_double, #alpha
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.float64,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    #prep zgemm.c inputs
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M

    # function call
    custom_functions.dgemmexp(M, N, K, a, A, ldA, B, ldB, C, ldC)

    return C

def call_zgemm(M,N,K,A,B, a = 1, b = 0):
    # setting up C function
    so_file = "/home/etolley/bluebild/pypeline/src/zgemm-splat.so"
    custom_functions = CDLL(so_file)
    custom_functions.zgemm.argtypes=[c_int, c_int, c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), #alpha
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int,
                                    ndpointer(dtype=np.complex128,ndim=1,flags='C'), # beta
                                    ndpointer(dtype=np.complex128,ndim=2,flags='F'), c_int]
    A = np.asfortranarray(A)
    B = np.asfortranarray(B)
    ldA = M
    ldB = K
    C = np.zeros( (M,N),dtype =np.complex128, order='F')
    ldC = M
    alpha = make_complexdouble_array(a)
    beta =  make_complexdouble_array(b)

    # function call
    faulthandler.enable()
    custom_functions.zgemm(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC)

    return C

# setting up inputs
M = 550
N = 248*124
K = 3
A = np.random.rand(M,K).astype(np.float64) 
B = np.random.rand(K,N).astype(np.float64)


t0 = time.process_time()
result_dgemm1 = call_dgemm(M,N,K,A,B, 1,0)
print("DGEMM  time:", time.process_time() - t0)

t1 = time.process_time()
result_matmul = np.matmul(A,B)
print("numpy matmul time:", time.process_time() - t1)

t0 = time.process_time()
result_dgemmexp = call_dgemmexp(M,N,K,A,B, 1)
print("DGEMM+exp time:", time.process_time() - t0)

t1 = time.process_time()
result_matmulexp = np.exp(1j*np.matmul(A,B))
print("numpy exp(matmul) time:", time.process_time() - t1)

t2 = time.process_time()
result_matmul1 = np.matmul(A,B)
result_matmulexpne = np.zeros(result_matmul1.shape,dtype=np.complex64)
ne.evaluate( "exp(A * B)",dict(A=1j, B=result_matmul1),out=result_matmulexpne,casting="same_kind",) 
print("numpy matmul + numexpr exp time:", time.process_time() - t2)



#A = np.random.rand(M,K).astype(np.complex128) 
#B = np.random.rand(K,N).astype(np.complex128)

#t0 = time.process_time()
#result_zgemm = call_zgemm(M,N,K,A,B, 1,0)
#print("ZGEMM  time:", time.process_time() - t0)

print( "Agreement between matmul and dgemm:", np.mean(result_matmul-result_dgemm1))
print( "Agreement between exp(matmul) and dgemmexp:", np.mean(result_matmulexp-result_dgemmexp))
print( "Agreement between matmul + numexpr exp and dgemmexp:", np.mean(result_matmulexpne-result_dgemmexp))

#print(result_matmul)
#print(result_matmulexp)
#print(result_matmulexpne)

