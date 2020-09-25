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
#define M_BLOCK_SIZE 200
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


void dgemm( const int M, const int N, const int K, const double alpha_, const double *A, const int lda, const double *B, const int ldb, const double beta_, double * __restrict__ C, const int ldc){
    double alpha = alpha_;
    double beta  = beta_;
    int ib, jb, kb;
    int i, j, k;

    double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
    double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
    double AB[M_BLOCK_SIZE*N_BLOCK_SIZE];

    double a00, a01, a02, a03, a04, a05, a06, a07;
    for( int kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
        for( int ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
            otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
#pragma vector always 
            for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
                double * pA = &Ab[0];
                double * pB = &B [0];
                //
                int i;
                for (i = 0; i < Mb - Mb%8; i = i + 8){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        a01 = 0.;
                        a02 = 0.;
                        a03 = 0.;
                        a04 = 0.;
                        a05 = 0.;
                        a06 = 0.;
                        a07 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
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
                            double b0 = *(pB + 0);    
			    //
                            a00 += a0*b0;    // c00 = A[0]*B[0]
                            a01 += a1*b0;    // c01 = A[1]*B[0]
                            a02 += a2*b0;    // c02 = A[2]*B[0]
                            a03 += a3*b0;    // c03 = A[3]*B[0]
                            a04 += a4*b0;    // c04 = A[4]*B[0]
                            a05 += a5*b0;    // c05 = A[5]*B[0]
                            a06 += a6*b0;    // c06 = A[6]*B[0]
                            a07 += a7*b0;    // c07 = A[7]*B[0]
                            //
                            pA += 8;
                            pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = alpha*a00;// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = alpha*a01;// + beta*c01;
                        C[(j + jb + 0)*ldc + i + ib + 2] = alpha*a02;// + beta*c02;
                        C[(j + jb + 0)*ldc + i + ib + 3] = alpha*a03;// + beta*c03;
                        C[(j + jb + 0)*ldc + i + ib + 4] = alpha*a04;// + beta*c04;
                        C[(j + jb + 0)*ldc + i + ib + 5] = alpha*a05;// + beta*c05;
                        C[(j + jb + 0)*ldc + i + ib + 6] = alpha*a06;// + beta*c05;
                        C[(j + jb + 0)*ldc + i + ib + 7] = alpha*a07;// + beta*c07;
                    }
                }
                //
                for (; i < Mb - Mb%4; i = i + 4){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1) {
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        a01 = 0.;
                        a02 = 0.;
                        a03 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double a0 = *(pA + 0);
                                double a1 = *(pA + 1);
                                double a2 = *(pA + 2);
                                double a3 = *(pA + 3);
    //
                                double b0 = *(pB + 0);
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                a01 += a1*b0;   // c00 = A[0]*B[0]
                                a02 += a2*b0;   // c00 = A[0]*B[0]
                                a03 += a3*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 4;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = alpha*a00;// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = alpha*a01;// + beta*c01;
                        C[(j + jb + 0)*ldc + i + ib + 2] = alpha*a02;// + beta*c02;
                        C[(j + jb + 0)*ldc + i + ib + 3] = alpha*a03;// + beta*c03;
                    }
                }
                //
                for (; i < Mb - Mb%2; i = i + 2){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        //
                        a00 = 0.;
                        a01 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double a0 = *(pA + 0);
                                double a1 = *(pA + 1);
    //
                                double b0 = *(pB + 0);
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                a01 += a1*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 2;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = alpha*a00;// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = alpha*a01;// + beta*c01;
                    }
                }
                //
                for (; i < Mb; i = i + 1){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double a0 = *(pA + 0);
                                double b0 = *(pB + 0);
    //
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 1;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = alpha*a00;// + beta*c00;
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
            otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
#pragma simd
            for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
                double * pA = &Ab[0];
                double * pB = &B [0];
                //
                int i = 0;
#if 1
                for (; i < Mb - Mb%8; i = i + 8){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        //PREFETCH2((void*) pB + 4);
                        //PREFETCH2((void*) pB + 8);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 0]);
                        //
                        a00 = 0.;
                        a01 = 0.;
                        a02 = 0.;
                        a03 = 0.;
                        a04 = 0.;
                        a05 = 0.;
                        a06 = 0.;
                        a07 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                            //PREFETCH0((void*) pB + 256);
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
                            double b0 = *(pB + 0);    
                            a00 += a0*b0;    // c00 = A[0]*B[0]
                            a01 += a1*b0;    // c01 = A[1]*B[0]
                            a02 += a2*b0;    // c02 = A[2]*B[0]
                            a03 += a3*b0;    // c03 = A[3]*B[0]
                            a04 += a4*b0;    // c04 = A[4]*B[0]
                            a05 += a5*b0;    // c05 = A[5]*B[0]
                            a06 += a6*b0;    // c06 = A[6]*B[0]
                            a07 += a7*b0;    // c07 = A[7]*B[0]
                            //
                            pA += 8;
                            pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(I*alpha*a00);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = cexp(I*alpha*a01);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 2] = cexp(I*alpha*a02);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 3] = cexp(I*alpha*a03);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 4] = cexp(I*alpha*a04);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 5] = cexp(I*alpha*a05);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 6] = cexp(I*alpha*a06);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 7] = cexp(I*alpha*a07);// + beta*c00;

                    }
                }
                //
                for (; i < Mb - Mb%4; i = i + 4){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1) {
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        //PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 0]);
                        //
                        a00 = 0.;
                        a01 = 0.;
                        a02 = 0.;
                        a03 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double a0 = *(pA + 0);
                                double a1 = *(pA + 1);
                                double a2 = *(pA + 2);
                                double a3 = *(pA + 3);
    //
                                double b0 = *(pB + 0);
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                a01 += a1*b0;   // c00 = A[0]*B[0]
                                a02 += a2*b0;   // c00 = A[0]*B[0]
                                a03 += a3*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 4;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(I*alpha*a00);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = cexp(I*alpha*a01);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 2] = cexp(I*alpha*a02);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 3] = cexp(I*alpha*a03);// + beta*c00;
                    }
                }
                //
                for (; i < Mb - Mb%2; i = i + 2){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        //PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 0]);
                        //
                        //
                        a00 = 0.;
                        a01 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double a0 = *(pA + 0);
                                double a1 = *(pA + 1);
    //
                                double b0 = *(pB + 0);
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                a01 += a1*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 2;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(I*alpha*a00);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = cexp(I*alpha*a01);// + beta*c01;
                    }
                }
#endif
                //
                for (; i < Mb; i = i + 1){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double * pB = &B[j*Kb + 0];
                        //
                        //PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 0]);
                        //
                        a00 = 0.;
                        //
                        double * pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double a0 = *(pA + 0);
                                double b0 = *(pB + 0);
    //
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 1;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(I*alpha*a00);// + beta*c00;
                    }
                }
            }
        } 
    }  
}
