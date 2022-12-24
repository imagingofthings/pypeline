#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
extern "C" {

typedef enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;

#ifdef BLUEBILD_BLAS_C
void cblas_sgemm(enum CBLAS_LAYOUT order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc);

void cblas_dgemm(enum CBLAS_LAYOUT order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, double alpha, const double *A, int lda, const double *B,
                 int ldb, double beta, double *C, int ldc);

void cblas_cgemm(enum CBLAS_LAYOUT order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, const void *alpha, const void *A, int lda, const void *B,
                 int ldb, const void *beta, void *C, int ldc);

void cblas_zgemm(enum CBLAS_LAYOUT order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, const void *alpha, const void *A, int lda, const void *B,
                 int ldb, const void *beta, void *C, int ldc);

void cblas_ssymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta, float *C, const int ldc);

void cblas_dsymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta, double *C, const int ldc);

void cblas_csymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta, void *C, const int ldc);

void cblas_zsymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const void *alpha, const void *A, const int lda,
                 const void *B, const int ldb, const void *beta, void *C, const int ldc);
#else


void sgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N,
            const int *K, const void *ALPHA, const void *A, const int *LDA,
            const void *B, const int *LDB, const void *BETA, void *C,
            const int *LDC, int TRANSA_len, int TRANSB_len);

void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N,
            const int *K, const void *ALPHA, const void *A, const int *LDA,
            const void *B, const int *LDB, const void *BETA, void *C,
            const int *LDC, int TRANSA_len, int TRANSB_len);

void cgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N,
            const int *K, const void *ALPHA, const void *A, const int *LDA,
            const void *B, const int *LDB, const void *BETA, void *C,
            const int *LDC, int TRANSA_len, int TRANSB_len);

void zgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N,
            const int *K, const void *ALPHA, const void *A, const int *LDA,
            const void *B, const int *LDB, const void *BETA, void *C,
            const int *LDC, int TRANSA_len, int TRANSB_len);

void ssymm_(const char *SIDE, const char *UPLO, const int *M, const int *N,
            const void *ALPHA, const void *A, const int *LDA, const void *B,
            const int *LDB, const void *BETA, void *C, const int *LDC,
            int SIDE_len, int UPLO_len);

void dsymm_(const char *SIDE, const char *UPLO, const int *M, const int *N,
            const void *ALPHA, const void *A, const int *LDA, const void *B,
            const int *LDB, const void *BETA, void *C, const int *LDC,
            int SIDE_len, int UPLO_len);

void csymm_(const char *SIDE, const char *UPLO, const int *M, const int *N,
            const void *ALPHA, const void *A, const int *LDA, const void *B,
            const int *LDB, const void *BETA, void *C, const int *LDC,
            int SIDE_len, int UPLO_len);

void zsymm_(const char *SIDE, const char *UPLO, const int *M, const int *N,
            const void *ALPHA, const void *A, const int *LDA, const void *B,
            const int *LDB, const void *BETA, void *C, const int *LDC,
            int SIDE_len, int UPLO_len);

#endif
}

namespace bluebild {
namespace blas {

inline auto cblas_transpose_to_string(CBLAS_TRANSPOSE op) -> const char * {
  if (op == CblasTrans)
    return "T";
  if (op == CblasConjTrans)
    return "C";
  return "N";
}

inline auto cblas_uplo_to_string(CBLAS_UPLO uplo) -> const char * {
  if (uplo == CblasUpper)
    return "U";
  return "L";
}

inline auto cblas_diag_to_string(CBLAS_DIAG diag) -> const char * {
  if (diag == CblasNonUnit)
    return "N";
  return "U";
}

inline auto cblas_side_to_string(CBLAS_SIDE side) -> const char * {
  if (side == CblasLeft)
    return "L";
  return "R";
}

inline auto gemm(CBLAS_LAYOUT order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N,
                 int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                 float *C, int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  sgemm_(cblas_transpose_to_string(transA), cblas_transpose_to_string(transB),
         &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto gemm(CBLAS_LAYOUT order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N,
                 int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta,
                 double *C, int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  dgemm_(cblas_transpose_to_string(transA), cblas_transpose_to_string(transB),
         &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto gemm(CBLAS_LAYOUT order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N,
                 int K, std::complex<float> alpha, const std::complex<float> *A, int lda,
                 const std::complex<float> *B, int ldb, std::complex<float> beta,
                 std::complex<float> *C, int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_cgemm(order, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
#else
  cgemm_(cblas_transpose_to_string(transA), cblas_transpose_to_string(transB),
         &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto gemm(CBLAS_LAYOUT order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N,
                 int K, std::complex<double> alpha, const std::complex<double> *A, int lda,
                 const std::complex<double> *B, int ldb, std::complex<double> beta,
                 std::complex<double> *C, int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_zgemm(order, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
#else
  zgemm_(cblas_transpose_to_string(transA), cblas_transpose_to_string(transB),
         &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto symm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta, float *C, const int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_ssymm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  ssymm_(cblas_side_to_string(Side), cblas_uplo_to_string(Uplo), &M, &N, &alpha,
         A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto symm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta, double *C, const int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_dsymm(layout, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  dsymm_(cblas_side_to_string(Side), cblas_uplo_to_string(Uplo), &M, &N, &alpha,
         A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto symm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const std::complex<float> alpha,
                 const std::complex<float> *A, const int lda, const std::complex<float> *B,
                 const int ldb, const std::complex<float> beta, std::complex<float> *C,
                 const int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_csymm(layout, Side, Uplo, M, N, &alpha, A, lda, B, ldb, &beta, C, ldc);
#else
  csymm_(cblas_side_to_string(Side), cblas_uplo_to_string(Uplo), &M, &N, &alpha,
         A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

inline auto symm(const CBLAS_LAYOUT layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
                 const int M, const int N, const std::complex<double> alpha,
                 const std::complex<double> *A, const int lda, const std::complex<double> *B,
                 const int ldb, const std::complex<double> beta, std::complex<double> *C,
                 const int ldc) -> void {
#ifdef BLUEBILD_BLAS_C
  cblas_zsymm(layout, Side, Uplo, M, N, &alpha, A, lda, B, ldb, &beta, C, ldc);
#else
  zsymm_(cblas_side_to_string(Side), cblas_uplo_to_string(Uplo), &M, &N, &alpha,
         A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
#endif
}

} // namespace blas
}  // namespace bluebild
