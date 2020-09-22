#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "otcopy_8.h"

//#include "otcopy_8.h"

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 200
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 3000
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


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* __restrict__ C, const int ldc)
{
	int ib, jb, kb;
	int i, j, k;
	//
	double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
	double AB[M_BLOCK_SIZE*N_BLOCK_SIZE];
	//
	double copytime    = 0.;
        double computetime = 0.;
	//
	__m256d t00, t01, t02, t03;
	__m256d t04, t05, t06, t07;
	__m256d t08, t09, t10, t11;
	__m256d t12, t13, t14, t15;
	//
	__m256d y00, y01, y02, y03;
	__m256d y04, y05, y06, y07;
	__m256d y08, y09, y10, y11;
	__m256d y12, y13, y14, y15;
	//
	__m128d x00, x01, x02, x03;
	__m128d x04, x05, x06, x07;
	__m128d x08, x09, x10, x11;
	__m128d x12, x13, x14, x15;
	//
	double c00, c01, c02, c03, c04, c05, c06, c07;
	for( int kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
		for( int ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
			otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
			for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
				computetime -= myseconds();
				double* pA = &Ab[0];
				double* pB = &B[0];

				for (int i = 0; i < Mb - Mb%8; i = i + 8){
	#pragma simd lastprivate(c00, c01, c02, c03, c04, c05, c06, c07)
					for (int j = 0; j < Nb - Nb%1; j = j + 1){
						//
                                                double* pB = &B[j*Kb + 0];
                                                PREFETCH2((void*) pB + 0);
						PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                                                //
						 c00 = C[(j + jb + 0)*ldc + i + ib + 0];
						 c01 = C[(j + jb + 0)*ldc + i + ib + 1];
						 c02 = C[(j + jb + 0)*ldc + i + ib + 2];
						 c03 = C[(j + jb + 0)*ldc + i + ib + 3];
						 c04 = C[(j + jb + 0)*ldc + i + ib + 4];
						 c05 = C[(j + jb + 0)*ldc + i + ib + 5];
						 c06 = C[(j + jb + 0)*ldc + i + ib + 6];
						 c07 = C[(j + jb + 0)*ldc + i + ib + 7];
						//
						double* pA = &Ab[i*Kb + 0];
						//
						for (int k = 0; k < 3; k = k + 1)
						{
							PREFETCH0((void*) pB + 256);
							//
							double a0 = *(pA + 0);
							double a1 = *(pA + 1);
							double a2 = *(pA + 2);
							double a3 = *(pA + 3);
							double a4 = *(pA + 4);
							double a5 = *(pA + 5);
							double a6 = *(pA + 6);
							double a7 = *(pA + 7);
							//
							double b0 = *(pB + 0);	// y02 = B[0]
							//double b1 = *(pB + 1);	// y02 = B[0]
							//double b2 = *(pB + 2);	// y02 = B[0]
							#if 1
							c00 += a0*b0;	// y06 = A[0]*B[0]
							c01 += a1*b0;	// y06 = A[0]*B[0]
							c02 += a2*b0;	// y06 = A[0]*B[0]
							c03 += a3*b0;	// y06 = A[0]*B[0]
							c04 += a4*b0;	// y06 = A[0]*B[0]
							c05 += a5*b0;	// y06 = A[0]*B[0]
							c06 += a6*b0;	// y06 = A[0]*B[0]
							c07 += a7*b0;	// y06 = A[0]*B[0]
							#else
							C[(j + jb + 0)*ldc + i + ib + 0] += a0*b0;
							C[(j + jb + 0)*ldc + i + ib + 1] += a1*b0;
							C[(j + jb + 0)*ldc + i + ib + 2] += a2*b0;
							C[(j + jb + 0)*ldc + i + ib + 3] += a3*b0;
							C[(j + jb + 0)*ldc + i + ib + 4] += a4*b0;
							C[(j + jb + 0)*ldc + i + ib + 5] += a5*b0;
							C[(j + jb + 0)*ldc + i + ib + 6] += a6*b0;
							C[(j + jb + 0)*ldc + i + ib + 7] += a7*b0;
							#endif
							pA += 8;
							pB += 1;
						}
						//
						#if 1
						C[(j + jb + 0)*ldc + i + ib + 0] = (c00);
						C[(j + jb + 0)*ldc + i + ib + 1] = (c01);
						C[(j + jb + 0)*ldc + i + ib + 2] = (c02);
						C[(j + jb + 0)*ldc + i + ib + 3] = (c03);
						C[(j + jb + 0)*ldc + i + ib + 4] = (c04);
						C[(j + jb + 0)*ldc + i + ib + 5] = (c05);
						C[(j + jb + 0)*ldc + i + ib + 6] = (c06);
						C[(j + jb + 0)*ldc + i + ib + 7] = (c07);
						#endif
					}
				}
				computetime += myseconds();
			}
		} //
	}
	printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
}
