#include <complex>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "host/nufft_3d3_host.hpp"

#include <finufft.h>

namespace bluebild {

template <>
Nufft3d3Host<double>::Nufft3d3Host(int iflag, double tol, int numTrans, int M,
                                   const double *x, const double *y,
                                   const double *z, int N, const double *s,
                                   const double *t, const double *u) {
  nufft_opts opts;
  finufft_default_opts(&opts);
  finufft_plan p;
  if (finufft_makeplan(3, 3, nullptr, iflag, numTrans, tol, &p, &opts))
    throw FiNUFFTError();

  plan_ = planType(new finufft_plan(p), [](void *ptr) {
    auto castPtr = reinterpret_cast<finufft_plan *>(ptr);
    finufft_destroy(*castPtr);
    delete castPtr;
  });

  finufft_setpts(p, M, const_cast<double *>(x), const_cast<double *>(y),
                 const_cast<double *>(z), N, const_cast<double *>(s),
                 const_cast<double *>(t), const_cast<double *>(u));
}

template <>
void Nufft3d3Host<double>::execute(const std::complex<double> *cj,
                                   std::complex<double> *fk) {
  finufft_execute(*reinterpret_cast<const finufft_plan *>(plan_.get()),
                  const_cast<std::complex<double> *>(cj), fk);
}

template <>
Nufft3d3Host<float>::Nufft3d3Host(int iflag, float tol, int numTrans, int M,
                                  const float *x, const float *y,
                                  const float *z, int N, const float *s,
                                  const float *t, const float *u) {
  nufft_opts opts;
  finufftf_default_opts(&opts);
  finufftf_plan p;
  if (finufftf_makeplan(3, 3, nullptr, iflag, numTrans, tol, &p, &opts))
    throw FiNUFFTError();

  plan_ = planType(new finufftf_plan(p), [](void *ptr) {
    auto castPtr = reinterpret_cast<finufftf_plan *>(ptr);
    finufftf_destroy(*castPtr);
    delete castPtr;
  });

  finufftf_setpts(p, M, const_cast<float *>(x), const_cast<float *>(y),
                  const_cast<float *>(z), N, const_cast<float *>(s),
                  const_cast<float *>(t), const_cast<float *>(u));
}

template <>
void Nufft3d3Host<float>::execute(const std::complex<float> *cj,
                                  std::complex<float> *fk) {
  finufftf_execute(*reinterpret_cast<const finufftf_plan *>(plan_.get()),
                   const_cast<std::complex<float> *>(cj), fk);
}

} // namespace bluebild
