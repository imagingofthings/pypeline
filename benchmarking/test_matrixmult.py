import numpy as np
import numexpr as ne
import timing
from math import exp


if __name__ == "__main__":

    # timer
    timer = timing.Timer()

    (N_antenna, N_height, N_width) = (550, 248, 124)

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

    diff = np.average( C[:,:,0] - C1)
    # if the dot product is done on each slice in the 124-sized dim, diff should be zero
    print(diff)

    ####################################################################
    # test numepr
    P = np.zeros( (N_antenna, N_height, N_width))
    ne.evaluate(
                "exp(A * B)",
                dict(A=2, B=C),
                out=P,
                casting="same_kind",
            )  # Due to limitations of NumExpr2

    # if numexpr is doing an element-by-element EXP,
    # the two expressions below should evaluate to the same value
    print(P[0,0,0])
    print(exp(2*C[0,0,0]))