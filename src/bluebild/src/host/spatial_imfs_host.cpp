#include "host/spatial_imfs_host.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <iostream> //EO

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T>
static T calc_pi_sinc(T a, T x) {
  return x ? std::sin(a * x) / x : T(3.14159265358979323846);
}

template <typename T>
auto spatial_imfs_host(ContextInternal& ctx, T wl,
                       const T* grid, std::size_t gd1, std::size_t gd2, std::size_t gd3,
                       int nlevel, int precision) -> void {

    std::cout << "@@@ C++ spatial_imfs_host" << std::endl;

    assert(nlevel > 0);
    assert(precision == 32 || precision == 64);
    assert(gd1 == 3);

    /**
  auto bufferBase = create_buffer<std::complex<T>>(ctx.allocators().host(), m * m);
  auto basePtr = bufferBase.get();

  auto x = xyz;
  auto y = xyz + ldxyz;
  auto z = xyz + 2 * ldxyz;
  T sincScale = 2 * 3.14159265358979323846 / wl;
  for(std::size_t i = 0; i < m;++i) {
    basePtr[i * m + i] = 4 * 3.14159265358979323846;
    for (std::size_t j = i + 1; j < m; ++j) {
      auto diffX = x[i] - x[j];
      auto diffY = y[i] - y[j];
      auto diffZ = z[i] - z[j];
      auto norm = std::sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
      basePtr[i * m + j] = 4 * calc_pi_sinc(sincScale, norm);
    }
  }

  auto bufferC = create_buffer<std::complex<T>>(ctx.allocators().host(), m * n);

  blas::symm(CblasColMajor, CblasLeft, CblasLower, m, n, {1, 0}, basePtr, m, w, ldw, {0, 0},
             bufferC.get(), m);
  blas::gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, m, {1, 0}, w, ldw, bufferC.get(), m,
             {0, 0}, g, ldg);
    **/
}

    template auto spatial_imfs_host<float>(ContextInternal& ctx, float wl,
                                           const float* grid, std::size_t gd1, std::size_t gd2, std::size_t gd3,
                                           int nlevel, int precision) -> void;
    
    template auto spatial_imfs_host<double>(ContextInternal& ctx, double wl,
                                            const double* grid, std::size_t gd1, std::size_t gd2, std::size_t gd3,
                                            int nlevel, int precision) -> void;
    
}  // namespace bluebild
