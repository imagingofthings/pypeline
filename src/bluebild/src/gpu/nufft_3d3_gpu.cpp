#include <complex>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "gpu/nufft_3d3_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

#include <cufinufft.h>

namespace bluebild {

template <>
Nufft3d3GPU<double>::Nufft3d3GPU(int iflag, double tol, int numTrans, int M,
                                 const double *x, const double *y,
                                 const double *z, int N, const double *s,
                                 const double *t, const double *u) {
  cufinufft_opts opts;
  cufinufft_default_opts(3, 3, &opts);
  cufinufft_plan p;
  if (cufinufft_makeplan(3, 3, nullptr, iflag, numTrans, tol, numTrans, &p,
                         &opts))
    throw FiNUFFTError();

  plan_ = planType(new cufinufft_plan(p), [](void *ptr) {
    auto castPtr = reinterpret_cast<cufinufft_plan *>(ptr);
    cufinufft_destroy(*castPtr);
    delete castPtr;
  });

  if (cufinufft_setpts(M, const_cast<double *>(x), const_cast<double *>(y),
                       const_cast<double *>(z), N, const_cast<double *>(s),
                       const_cast<double *>(t), const_cast<double *>(u), p))
    throw FiNUFFTError();
}

template <>
void Nufft3d3GPU<double>::execute(const gpu::ComplexType<double> *cj,
                                  gpu::ComplexType<double> *fk) {
  cufinufft_execute(const_cast<gpu::ComplexType<double> *>(cj), fk,
                    *reinterpret_cast<const cufinufft_plan *>(plan_.get()));
}

template <>
Nufft3d3GPU<float>::Nufft3d3GPU(int iflag, float tol, int numTrans, int M,
                                const float *x, const float *y, const float *z,
                                int N, const float *s, const float *t,
                                const float *u) {
  cufinufft_opts opts;
  cufinufftf_default_opts(3, 3, &opts);
  cufinufftf_plan p;
  if (cufinufftf_makeplan(3, 3, nullptr, iflag, numTrans, tol, numTrans, &p,
                          &opts))
    throw FiNUFFTError();

  plan_ = planType(new cufinufftf_plan(p), [](void *ptr) {
    auto castPtr = reinterpret_cast<cufinufftf_plan *>(ptr);
    cufinufftf_destroy(*castPtr);
    delete castPtr;
  });

  if (cufinufftf_setpts(M, const_cast<float *>(x), const_cast<float *>(y),
                        const_cast<float *>(z), N, const_cast<float *>(s),
                        const_cast<float *>(t), const_cast<float *>(u), p))
    throw FiNUFFTError();
}

template <>
void Nufft3d3GPU<float>::execute(const gpu::ComplexType<float> *cj,
                                 gpu::ComplexType<float> *fk) {
  cufinufftf_execute(const_cast<gpu::ComplexType<float> *>(cj), fk,
                     *reinterpret_cast<const cufinufftf_plan *>(plan_.get()));
}

} // namespace bluebild
