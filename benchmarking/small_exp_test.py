import numpy as np
import numexpr as ne

a = np.random.rand(40,40,40) + 1j*np.random.rand(40,40,40)
a = a.astype(np.complex128)
b1 = np.zeros((40,40,40)).astype(np.complex128)
ne.evaluate("exp(A* B)", dict(A=2, B=a), out=b1, casting="same_kind")

b2 = np.exp(2*a)

print( np.average(b1-b2))