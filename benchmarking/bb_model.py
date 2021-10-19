import math
def run_eigenvalue_decomposition(Nstation):
    # S dimension is Nstation x Nstation
    # G dimension is Nstation x Nstation

    #print("Calculating comlexity for eigenvalue decompositon...")

    # eigh complexity is O(N^3), so Nstation^3
    #Ds, Vs = scipy.linalg.eigh(S)
    #print("scipy.linalg.eigh( [{0},{0}]) => {1}".format(Nstation, Nstation**3))

    # matmul comlexity is O(N^3), so Nstation^3
    #A = (Vs * Ds) @ Vs.conj().T
    #print("matmuls of [{0},{0}]x[{0},{0}] => {1}".format(Nstation, Nstation**3))

    # eigh complexity is O(N^3), so Nstation^3
    #D, V = scipy.linalg.eigh(A, G)
    #print("scipy.linalg.eigh( [{0},{0}]) => {1}".format(Nstation, Nstation**3))

    return 3*Nstation**3

def run_nufft2d(T, Nx, Ny, Nstation, Neig, Nantenna):
    o = 0
    
    # start with:
    # XYZ dimension is 3xNantenna
    # W dimension is Nantenna, Nstation
    # G dimension is Nstation x Nstation
    # S dimension is Nstation x Nstation

    #== inside T loop ==============
    # V dimension is Nstation v Neig
    # D dimension is Neig x Neig
    o += T*run_eigenvalue_decomposition(Nstation)

    # matmul to calculate UVW
    o += 3*3*Nantenna

    # construct UVW_baselines
    o += Nantenna*Nantenna*3

    #scale baselines
    o += Nantenna*Nantenna*3

    # calculate S_corrected
    ## matmul V*D => Nstation x Neig
    o += Nstation * Neig * Neig
    ## matmul [Nstation x Neig]x[Nstation v Neig] =>  
    o += Nstation*Nstation*Neig
    ## matmul [Nstation, Nstation] x [ Nantenna, Nstation]
    o += Nstation*Nstation*Nantenna
    ## matmul [Nantenna, Nstation] x [ Nantenna, Nstation]
    o += Nstation*Nantenna*Nantenna


    #== end T loop  ==============
    o*=T

    # w correction
    o += Nantenna*Nantenna*Nantenna

    # fft is NlogN for N fourier modes from N data points
    # N data points = Nantenna*Nantenna
    # N modes = Nx*Ny
    o += Nantenna*Nantenna*math.log2(Nx*Ny)

    #take real part
    o +=  Nx*Ny

    return o

def run_ss(T, Nx, Ny, Nstation, Neig, Nantenna):
    o = 0
    
    # start with:
    # XYZ dimension is 3xNantenna
    # W dimension is Nantenna, Nstation
    # G dimension is Nstation x Nstation
    # S dimension is Nstation x Nstation
    # grid is 3 x Nx x Ny

    #== inside T loop ==============
    # D, V = pylinalg.eigh(S, G, tau=1, N=self._N_eig)
    # V dimension is Nstation v Neig
    # D dimension is Neig x Neig
    o += run_eigenvalue_decomposition(Nstation)

    # Standard synthesis kernel
    #     loop over Nx
    #     matmul: [3,Nantenna]x[3,Ny]
    oi = 3*Nantenna*Ny
    #     multiply with complex number
    oi += Nantenna*Ny
    #     pointwise exponential
    oi += Nantenna*Ny
    #     multiply with W, output is Ny*Nstation
    oi += Nantenna*Ny*Nstation
    #     multiply with V, output is Ny*Eig
    oi += Ny*Nstation*Neig
    #     end loop
    o += Nx*oi
    # normalize
    o += Nx*Ny*Neig

    # rescale energy eigenlevels

    o += Nx*Ny*Neig*Neig

    #== end T loop  ==============
    o *= T

    return o

if __name__ == "__main__":
    T = 72
    Nx, Ny = 496,248
    Nstation, Nantenna = 24, 550
    Neig = 14

    print('''Running with parameters:\n\t{0} timesteps \n\t{1} Stations \n\t{2} Antennas \n\t{3} Eigenimages \n\t{4} Pixels in output image
        '''.format(T, Nstation, Nantenna, Neig, Nx*Ny  ))

    o_ss = run_ss(T, Nx, Ny, Nstation, Neig, Nantenna )
    print("N ops for standard synthesis: {0}\n".format(o_ss))
    print("Gflops for standard synthesis: {0}\n".format(o_ss*2/1e9))
    o_nufft = run_nufft2d(T, Nx, Ny, Nstation, Neig, Nantenna )
    print("N ops for NUFFT synthesis: {0}\n".format(o_nufft))
    print("Gflops for NUFFT synthesis: {0}\n".format(o_nufft*2/1e9))
