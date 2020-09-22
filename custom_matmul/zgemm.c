//author gilles.fourestey@epfl.ch

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <float.h>
#include <math.h>
#include <complex.h>

#include <sys/types.h>
#include <sys/resource.h>

#include <unistd.h>
#include <sys/time.h>

#include "mm_malloc.h"
//#include <mkl.h>

#ifdef PAPI
#include "cscs_papi.h"
#endif

#define _alpha 1.e0 + I*1.e0
#define _beta  1.e0 + I*1.e0

#define NN 32
#define NRUNS 1

//#define PAPI

/*
 *   Your function _MUST_ have the following signature:
 */


double mysecond()
{
	struct timeval  tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


static void init(double complex *A, int M, int N, double complex v)
{
        int ii, jj;
        for (ii = 0; ii < M; ++ii)
                for (jj= 0; jj < N; ++jj)
                        A[jj*M + ii] = v + jj + 0*(jj*N + ii + 0);
}

static void init_t(double complex *A, int M, int N, double complex v)
{
        int ii, jj;
        for (ii = 0; ii < M; ++ii)
                for (jj= 0; jj < N; ++jj)
                        A[jj*M + ii] = ii + 0*(ii + 0);
}


void 
matrix_init (double complex *A, const int M, const int N, double complex val)
{
	int i;

	for (i = 0; i < M*N; ++i) 
	{
		A[i] = creal(val)*drand48() + cimag(val)*I*drand48();
		//A[i] = val;
		//A[i] = 1.;
	}
}

void 
matrix_clear (double *C, const int M, const int N) 
{
	memset (C, 0, M*N*sizeof (double));
}


double
time_zgemm (const int M, const int N, const unsigned K,
		const double complex alpha, const double complex *A, const int lda,
		const double complex *B, const int ldb,
		const double complex beta, double complex *C, const unsigned ldc)
{
	double mflops, mflop_s;
	double secs = -1;

	int num_iterations = NRUNS;
	int i;


	double cpu_time = 0;

	double last_clock = mysecond();
	for (i = 0; i < num_iterations; ++i) 
	{
		cpu_time -= mysecond();
		zgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
		cpu_time += mysecond();
	}

	mflops  = 2.0*num_iterations*M*N*K/1.0e6;
	secs    = cpu_time;
	mflop_s = mflops/secs;


	//memcpy(C, Ca, N*ldc*sizeof(double complex));
#ifdef PAPI
	PAPI_FLUSH;
#endif
	return mflop_s;
}



double time_zgemm_blas(const int M, const unsigned N, const int K,
		const double alpha, const double complex *A, const int lda, 
		const double complex *B, const int ldb,
		const double beta, double complex *C, const int ldc)
{

	double mflops, mflop_s;
	double secs = -1;

	int num_iterations = NRUNS;
	int i;

	char transa = 'n';
	char transb = 'n';

	//double complex* Ca = (double complex*) _mm_malloc(N*ldc*sizeof(double complex), 32);

	double cpu_time = 0;

	for (i = 0; i < num_iterations; ++i)
	{
		//memcpy(Ca, C, N*ldc*sizeof(double complex));
		cpu_time -= mysecond();	
		zgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
		cpu_time += mysecond();
	}

	mflops  = 2.0*num_iterations*M*N*K/1.0e6;
	secs    = cpu_time;
	mflop_s = mflops/secs;

	//memcpy(C, Ca, N*ldc*sizeof(double complex));
#ifdef PAPI
	PAPI_FLUSH;
#endif
	return mflop_s;
}

	int
main (int argc, char *argv[])
{
	int sz_i;
	double mflop_s, mflop_b;

	int M, N, K;

	if ( argc == 4 )
	{
		M    = atoi(argv[1]);
		N    = atoi(argv[2]);
		K    = atoi(argv[3]);
	}
	else if (argc == 2)
	{
		M    = atoi(argv[1]);
		N = M;
		K = M;
	}
	else
	{
		M = NN;
		N = NN;
		K = NN;
	}

	int lda = M;
	int ldb = K;
	int ldc = M;

	double complex z1 = 1.0 + 2.0*I;
	double complex z2 = 2.0 + 1.0*I;
	double complex z3 = z1*z2;

	double alpha = 1. + I*0.0;
	double beta  = 0. + I*0.0;

	double complex* A  = (double complex*) _mm_malloc(M*K*sizeof(double complex), 32);
	double complex* B  = (double complex*) _mm_malloc(K*N*sizeof(double complex), 32);
	double complex* C  = (double complex*) _mm_malloc(M*N*sizeof(double complex), 32);
	double complex* Cb = (double complex*) _mm_malloc(M*N*sizeof(double complex), 32);

#if 1
	double complex one  = 1. + 1.*I;
	double complex zero = 0. + 0.*I;
	matrix_init(A,  M, K, one);
	//init_t(A,  M, K, 1.);
	matrix_init(B,  K, N, 2.*one);
	//init_t(B,  K, N, 1.);
	//matrix_clear(C);
	matrix_init(C,  M, N, one);
	memcpy(Cb, C, M*N*sizeof(double complex));
	//matrix_init(Cb, M, N, 0.);
#endif

	//const int M = test_sizes[sz_i];
	printf("Size: %u %u %u \t", M, N, K); fflush(stdout);

	mflop_s = time_zgemm     (M, N, K, alpha, A, lda, B, ldb, beta, C , ldc);    
	printf ("Gflop/s: %g ", mflop_s/1000.); fflush(stdout);
#if 1
	mflop_b = time_zgemm_blas(M, N, K, alpha, A, lda, B, ldb, beta, Cb, ldc);
	printf ("blas Gflops: %g\n", mflop_b/1000.);
#if 1
	printf("%f %f = %f %f\n", creal(C[0]), cimag(C[0]), creal(Cb[0]), cimag(Cb[0]));
	int ii, jj;
	double complex norm = 0. + 0.*I;
	for (jj = 0; jj < N; ++jj)
	{	
		for (ii = 0; ii < M; ++ii)
		{
			norm += csqrt(cabs(C[jj*ldc + ii] - Cb[jj*ldc + ii]));
			if (cabs(C[jj*ldc + ii] - Cb[jj*ldc + ii]) > 1e-14)
				printf("i = %d, j = %d, C = %f %f, should be %f %f, %.15f\n", ii, jj, creal(C[jj*ldc + ii]), cimag(C[jj*ldc + ii]), creal(Cb[jj*ldc + ii]), cimag(Cb[jj*ldc + ii]), cabs(C[jj*ldc + ii] - Cb[jj*ldc + ii]));
		}
	} 
	printf("||zgemm - blas||_l2 = %f\n", norm);
#endif
#endif
	printf("\n");
	//
	_mm_free(A);
	_mm_free(B);
	_mm_free(C);
	_mm_free(Cb);
	//
	return 0;
}
