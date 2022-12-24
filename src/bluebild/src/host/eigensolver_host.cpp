#include "host/eigensolver_host.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/lapack_api.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"

namespace bluebild {
template <typename T>
static auto copy_2d(std::size_t m, std::size_t n, const T* a, std::size_t lda, T* b,
                    std::size_t ldb) {
  if(lda == ldb && lda == m) {
    std::memcpy(b, a, m * n * sizeof(T));
  } else {
    for(std::size_t i = 0; i < n; ++i) {
      std::memcpy(b + i * ldb, a + i * lda, m * sizeof(T));
    }
  }
}

template <typename T>
auto eigh_host(ContextInternal& ctx, std::size_t m, std::size_t nEig, const std::complex<T>* a,
               std::size_t lda, const std::complex<T>* b, std::size_t ldb, int* nEigOut, T* d,
               std::complex<T>* v, std::size_t ldv) -> void {
  // copy input into buffer since eigensolver will overwrite
  auto bufferA = create_buffer<std::complex<T>>(ctx.allocators().host(), m * m);
  copy_2d(m, m, a, lda, bufferA.get(), m);

  auto bufferV = create_buffer<std::complex<T>>(ctx.allocators().host(), m * m);
  auto bufferD = create_buffer<T>(ctx.allocators().host(), m);
  auto bufferIfail = create_buffer<int>(ctx.allocators().host(), m);

  auto bufferPtrD = bufferD.get();
  auto bufferPtrV = bufferV.get();

  int hMeig = 0;
  if (lapack::eigh_solve(LapackeLayout::COL_MAJOR, 'V', 'V', 'L', m, bufferA.get(), m,
                         std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::max(), 0, m - 1,
                         &hMeig, bufferD.get(), bufferV.get(), m, bufferIfail.get())) {
    throw EigensolverError();
  }

  if(b) {
    if(hMeig != m) {
      // reconstruct A from positive eigenvalues only

      auto bufferC = create_buffer<std::complex<T>>(ctx.allocators().host(), hMeig * m);
      auto bufferPtrC = bufferC.get();

      // Compute C=V * diag(D)
      for (std::size_t i = 0; i < static_cast<std::size_t>(hMeig); ++i) {
        auto a = bufferPtrD[i];
        for (std::size_t j = 0; j < m; ++j) {
          bufferPtrC[i * m + j] = a * bufferPtrV[i * m + j];
        }
      }

      // Compute C = V * V^H
      blas::gemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, m, hMeig, std::complex<T>(1, 0),
                 bufferPtrC, m, bufferPtrC, m, std::complex<T>{0, 0}, bufferA.get(), m);

    } else {
      copy_2d(m, m, a, lda, bufferA.get(), m);
    }
    // copy input into buffer since eigensolver will overwrite
    auto bufferB = create_buffer<std::complex<T>>(ctx.allocators().host(), m * m);
    copy_2d(m, m, b, ldb, bufferB.get(), m);

    if (lapack::eigh_solve(LapackeLayout::COL_MAJOR, 1, 'V', 'V', 'L', m, bufferA.get(), m,
                           bufferB.get(), m, std::numeric_limits<T>::epsilon(),
                           std::numeric_limits<T>::max(), 0, m - 1, &hMeig, bufferD.get(),
                           bufferV.get(), m, bufferIfail.get())) {
      throw EigensolverError();
    }
  }

  // reorder into descending order. At most nEig eigenvalues / eigenvectors.
  for(std::size_t i = 0; i < std::min<std::size_t>(hMeig, nEig); ++i) {
    d[i] = bufferPtrD[hMeig - i - 1];
    std::memcpy(v + i * ldv, bufferPtrV + (hMeig - i - 1) * m, m * sizeof(std::complex<T>));
  }
  for(std::size_t i = hMeig; i < nEig; ++i) {
    d[i] = 0;
    std::memset(v + i * ldv, 0, m * sizeof(std::complex<T>));
  }

  *nEigOut = std::min<int>(hMeig, nEig);
}

template auto eigh_host<float>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                               const std::complex<float>* a, std::size_t lda,
                               const std::complex<float>* b, std::size_t ldb, int* nEigOut,
                               float* d, std::complex<float>* v, std::size_t ldv) -> void;

template auto eigh_host<double>(ContextInternal& ctx, std::size_t m, std::size_t nEig,
                                const std::complex<double>* a, std::size_t lda,
                                const std::complex<double>* b, std::size_t ldb, int* nEigOut,
                                double* d, std::complex<double>* v, std::size_t ldv) -> void;
}  // namespace bluebild
