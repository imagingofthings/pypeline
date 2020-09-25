#ifdef __GNUC__ 
#include <immintrin.h>
#include <complex.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "ztcopy_8.h"

//#include "otcopy_8.h"

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 100000
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 1000000
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 100000
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


void printc(double complex c)
{
	printf("z = %f %f\n", creal(c), cimag(c));
}


void zgemm( const int M, const int N, const int K, const double complex * alpha_, const double complex *A, const int lda, const double complex *B, const int ldb, const double complex * beta_, double complex* __restrict__ C, const int ldc){
    double complex alpha = alpha_[0];
    double complex beta  = beta_[0];
    int ib, jb, kb;
    int i, j, k;

    double complex Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
    double complex Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
    double complex AB[M_BLOCK_SIZE*N_BLOCK_SIZE];

    double complex a00, a01, a02, a03, a04, a05, a06, a07;
    for( int kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
        for( int ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
            otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
            for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
                double complex* pA = &Ab[0];
                double complex* pB = &B [0];
                //
                int i;
                for (i = 0; i < Mb - Mb%8; i = i + 8){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double complex* pB = &B[j*Kb + 0];
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
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                            PREFETCH0((void*) pB + 256);
                            //
                            double complex a0 = *(pA + 0);
                            double complex a1 = *(pA + 1);
                            double complex a2 = *(pA + 2);
                            double complex a3 = *(pA + 3);
                            double complex a4 = *(pA + 4);
                            double complex a5 = *(pA + 5);
                            double complex a6 = *(pA + 6);
                            double complex a7 = *(pA + 7);
                            //
                            double complex b0 = *(pB + 0);    
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
                        double complex* pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        a01 = 0.;
                        a02 = 0.;
                        a03 = 0.;
                        //
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double complex a0 = *(pA + 0);
                                double complex a1 = *(pA + 1);
                                double complex a2 = *(pA + 2);
                                double complex a3 = *(pA + 3);
    //
                                double complex b0 = *(pB + 0);
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
                        double complex* pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        //
                        a00 = 0.;
                        a01 = 0.;
                        //
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double complex a0 = *(pA + 0);
                                double complex a1 = *(pA + 1);
    //
                                double complex b0 = *(pB + 0);
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
                        double complex* pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        //
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double complex a0 = *(pA + 0);
                                double complex b0 = *(pB + 0);
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


void zgemmexp( const int M, const int N, const int K, const double complex * alpha_, const double complex *A, const int lda, const double complex *B, const int ldb, double complex* __restrict__ C, const int ldc){
    double complex alpha = alpha_[0];
    int ib, jb, kb;
    int i, j, k;

    double complex Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
    double complex Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
    double complex AB[M_BLOCK_SIZE*N_BLOCK_SIZE];

    double complex a00, a01, a02, a03, a04, a05, a06, a07;
    for( int kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
        for( int ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
            otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
            for( int jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
                double complex* pA = &Ab[0];
                double complex* pB = &B [0];
                //
                int i;
                for (i = 0; i < Mb - Mb%8; i = i + 8){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double complex* pB = &B[j*Kb + 0];
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
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                            PREFETCH0((void*) pB + 256);
                            //
                            double complex a0 = *(pA + 0);
                            double complex a1 = *(pA + 1);
                            double complex a2 = *(pA + 2);
                            double complex a3 = *(pA + 3);
                            double complex a4 = *(pA + 4);
                            double complex a5 = *(pA + 5);
                            double complex a6 = *(pA + 6);
                            double complex a7 = *(pA + 7);
                            //
                            double complex b0 = *(pB + 0);    
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
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(alpha*a00);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = cexp(alpha*a01);// + beta*c01;
                        C[(j + jb + 0)*ldc + i + ib + 2] = cexp(alpha*a02);// + beta*c02;
                        C[(j + jb + 0)*ldc + i + ib + 3] = cexp(alpha*a03);// + beta*c03;
                        C[(j + jb + 0)*ldc + i + ib + 4] = cexp(alpha*a04);// + beta*c04;
                        C[(j + jb + 0)*ldc + i + ib + 5] = cexp(alpha*a05);// + beta*c05;
                        C[(j + jb + 0)*ldc + i + ib + 6] = cexp(alpha*a06);// + beta*c05;
                        C[(j + jb + 0)*ldc + i + ib + 7] = cexp(alpha*a07);// + beta*c07;
                    }
                }
                //
                for (; i < Mb - Mb%4; i = i + 4){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1) {
                        //
                        double complex* pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        a01 = 0.;
                        a02 = 0.;
                        a03 = 0.;
                        //
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double complex a0 = *(pA + 0);
                                double complex a1 = *(pA + 1);
                                double complex a2 = *(pA + 2);
                                double complex a3 = *(pA + 3);
    //
                                double complex b0 = *(pB + 0);
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                a01 += a1*b0;   // c00 = A[0]*B[0]
                                a02 += a2*b0;   // c00 = A[0]*B[0]
                                a03 += a3*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 4;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(alpha*a00);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = cexp(alpha*a01);// + beta*c01;
                        C[(j + jb + 0)*ldc + i + ib + 2] = cexp(alpha*a02);// + beta*c02;
                        C[(j + jb + 0)*ldc + i + ib + 3] = cexp(alpha*a03);// + beta*c03;
                    }
                }
                //
                for (; i < Mb - Mb%2; i = i + 2){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double complex* pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        //
                        a00 = 0.;
                        a01 = 0.;
                        //
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double complex a0 = *(pA + 0);
                                double complex a1 = *(pA + 1);
    //
                                double complex b0 = *(pB + 0);
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                a01 += a1*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 2;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(alpha*a00);// + beta*c00;
                        C[(j + jb + 0)*ldc + i + ib + 1] = cexp(alpha*a01);// + beta*c01;
                    }
                }
                //
                for (; i < Mb; i = i + 1){
                    for (int j = 0; j < Nb - Nb%1; j = j + 1){
                        //
                        double complex* pB = &B[j*Kb + 0];
                        //
                        PREFETCH2((void*) pB + 0);
                        PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
                        //
                        a00 = 0.;
                        //
                        double complex* pA = &Ab[i*Kb + 0];
                        //
                        for (int k = 0; k < 3; k = k + 1)
                        {
                                double complex a0 = *(pA + 0);
                                double complex b0 = *(pB + 0);
    //
                                a00 += a0*b0;   // c00 = A[0]*B[0]
                                //
                                pA += 1;
                                pB += 1;
                        }
                        C[(j + jb + 0)*ldc + i + ib + 0] = cexp(alpha*a00);// + beta*c00;
                    }
                }
            }
        } 
    }  
}