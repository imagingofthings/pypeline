#ifdef __GNUC__ 
#include <immintrin.h>
#include <complex.h>
#include <math.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "otcopy_8.h"

//#include "otcopy_8.h"

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 2000
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 300000
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 200
#endif
#else
#define N_BLOCK_SIZE BLOCK_SIZE
#define M_BLOCK_SIZE BLOCK_SIZE
#define K_BLOCK_SIZE BLOCK_SIZE
#endif

#define PAGESIZE 4096;
#define NUMPERPAGE 512 // # of elements to fit a page


#define PREFETCH(A)  _mm_prefetch(A, _MM_HINT_NTA)
#define PREFETCH0(A) _mm_prefetch(A, _MM_HINT_T0)
#define PREFETCH1(A) _mm_prefetch(A, _MM_HINT_T1)
#define PREFETCH2(A) _mm_prefetch(A, _MM_HINT_T2)


#define min(a,b) (((a)<(b))?(a):(b))

#define STORE128(A, B) _mm_store_pd(A, B)
#define STORE256(A, B) _mm256_store_pd(A, B)


double myseconds()
{
        struct timeval  tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void print256(__m256d val)
{
	double a[4] = {0., 0., 0., 0.};
	_mm256_store_pd(&a[0], val);
	printf("%f %f %f %f", a[0], a[1], a[2], a[3]);

}

void print128(__m128d val)
{
        double a[2];
        _mm_store_pd(&a[0], val);
        printf("%f %f", a[0], a[1]);

}


void printc(double c)
{
	printf("z = %f\n", c);
}



void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double * __restrict__ C, const int ldc){
    int ib, jb, kb;
    int i, j, k;

    double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
    double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
    double AB[M_BLOCK_SIZE*N_BLOCK_SIZE];

    double a00, a01, a02, a03, a04, a05, a06, a07;
    for( int kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
	    for( int ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
		    //otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
#pragma vector always 
		    for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
			    //
			    double * pA = &A[0];
			    double * pB = &B[0];
			    //
			    for (int j = 0; j < Nb - Nb%1; j = j + 1)
			    {
				    //
				    //
				    PREFETCH2((void*) pB + 0);
				    PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
				    //
				    a00 = 0.;
				    //
				    double * pB = &B[j*Kb + 0];
				    //
				    for (i = 0; i < Mb; i = i + 1)
				    {
					    double *pA = &A[i];
					    //
					    double a0 = A[i + 0*lda];
					    double a1 = A[i + 1*lda];
					    double a2 = A[i + 2*lda];
					    //
					    double b0 = *(pB + 0);
					    double b1 = *(pB + 1);
					    double b2 = *(pB + 2);
						//
					    C[(j + jb)*ldc + i + ib] = alpha*(a0*b0 + a1*b1 + a2*b2);// + beta*c00;
				    }
			    }
		    }
	    } 
    }  
}


void dgemmexp( const int M, const int N, const int K, const double alpha, const double * __restrict__ A, const int lda, const double * __restrict__ B, const int ldb, double complex* __restrict__ C, const int ldc){
    int ib, jb, kb;
    int i, j, k;

    double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
    double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
    double AB[M_BLOCK_SIZE*N_BLOCK_SIZE];

    double a00, a01, a02, a03, a04, a05, a06, a07;
    for( int kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
	    for( int ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
		    //otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
#pragma vector always 
		    for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
			    //
			    double * pA = &A[0];
			    double * pB = &B[0];
			    //
			    for (int j = 0; j < Nb - Nb%1; j = j + 1)
			    {
				    //
				    //
				    PREFETCH2((void*) pB + 0);
				    PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
				    //
				    a00 = 0.;
				    //
				    double * pB = &B[j*Kb + 0];
				    //
				    for (i = 0; i < Mb; i = i + 1)
				    {
					    double *pA = &A[i];
					    //
					    double a0 = A[i + 0*lda];
					    double a1 = A[i + 1*lda];
					    double a2 = A[i + 2*lda];
					    //
					    double b0 = *(pB + 0);
					    double b1 = *(pB + 1);
					    double b2 = *(pB + 2);
						//
					    C[(j + jb)*ldc + i + ib] = cexpf(I*alpha*(a0*b0 + a1*b1 + a2*b2));// + beta*c00;
				    }
			    }
		    }
	    } 
    }  
}
