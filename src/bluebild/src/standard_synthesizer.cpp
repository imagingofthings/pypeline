#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "host/standard_synthesizer_host.hpp"


namespace bluebild {

template <typename T, typename>
BLUEBILD_EXPORT auto standard_synthesizer(Context &ctx,
                                          const T* d,
                                          const std::complex<T>* v,
                                          const T* xyz,
                                          const std::complex<T>* w,
                                          const size_t* c_idx,
                                          const size_t Nl,
                                          const T* grid,
                                          const T wl,
                                          const size_t Na,
                                          const size_t Nb,
                                          const size_t Nc,
                                          const size_t Ne,
                                          const size_t Nh,
                                          const size_t Nw,
                                          T* stats_std,
                                          T* stats_lsq,
                                          T* stats_std_cum,
                                          T* stats_lsq_cum) -> void {
    auto &ctxInternal = *InternalContextAccessor::get(ctx);
    if(ctx.processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        throw NotImplementedError();
#else
        throw GPUSupportError();
#endif
    } else {
        standard_synthesizer_host<T>(ctxInternal, d, v, xyz, w, c_idx, Nl, grid, wl,
                                     Na, Nb, Nc, Ne, Nh, Nw,
                                     stats_std, stats_lsq, stats_std_cum, stats_lsq_cum);
    }   
}

template auto standard_synthesizer(Context &ctx,
                                   const float* d,
                                   const std::complex<float>* v,
                                   const float* xyz,
                                   const std::complex<float>* w,
                                   const size_t* c_idx,
                                   const size_t Nl,
                                   const float* grid,
                                   const float wl,
                                   const size_t Na,
                                   const size_t Nb,
                                   const size_t Nc,
                                   const size_t Ne,
                                   const size_t Nh,
                                   const size_t Nw,
                                   float* stats_std,
                                   float* stats_lsq,
                                   float* stats_std_cum,
                                   float* stats_lsq_cum) -> void;

template auto standard_synthesizer(Context &ctx,
                                   const double* d,
                                   const std::complex<double>* v,
                                   const double* xyz,
                                   const std::complex<double>* w,
                                   const size_t* c_idx,
                                   const size_t Nl,
                                   const double* grid,
                                   const double wl,
                                   const size_t Na,
                                   const size_t Nb,
                                   const size_t Nc,
                                   const size_t Ne,
                                   const size_t Nh,
                                   const size_t Nw,
                                   double* stats_std,
                                   double* stats_lsq,
                                   double* stats_std_cum,
                                   double* stats_lsq_cum) -> void;

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_standard_synthesizer_s(BluebildContext ctx,
                                                              const float* d,
                                                              const void*  v,
                                                              const float* xyz,
                                                              const void*  w,
                                                              const size_t* c_idx,
                                                              const size_t  Nl,
                                                              const float* grid,
                                                              const float  wl,
                                                              const size_t Na,
                                                              const size_t Nb,
                                                              const size_t Nc,
                                                              const size_t Ne,
                                                              const size_t Nh,
                                                              const size_t Nw,
                                                              float* stats_std,
                                                              float* stats_lsq,
                                                              float* stats_std_cum,
                                                              float* stats_lsq_cum) {
    if (!ctx) {
        return BLUEBILD_INVALID_HANDLE_ERROR;
    }
    try {
        standard_synthesizer<float>(*reinterpret_cast<Context *>(ctx),
                                    d, 
                                    reinterpret_cast<const std::complex<float>*>(v),
                                    xyz,
                                    reinterpret_cast<const std::complex<float>*>(w),
                                    c_idx,
                                    Nl,
                                    grid,
                                    wl,
                                    Na, Nb, Nc, Ne, Nh, Nw,
                                    stats_std, stats_lsq, stats_std_cum, stats_lsq_cum);
    } catch (const bluebild::GenericError& e) {
        return e.error_code();
    } catch (...) {
        return BLUEBILD_UNKNOWN_ERROR;
    }
    return BLUEBILD_SUCCESS;
}
        
BLUEBILD_EXPORT BluebildError bluebild_standard_synthesizer_d(BluebildContext ctx,
                                                              const double* d,
                                                              const void* v,
                                                              const double* xyz,
                                                              const void* w,
                                                              const size_t* c_idx,
                                                              const size_t Nl,
                                                              const double* grid,
                                                              const double  wl,
                                                              const size_t Na,
                                                              const size_t Nb,
                                                              const size_t Nc,
                                                              const size_t Ne,
                                                              const size_t Nh,
                                                              const size_t Nw,
                                                              double* stats_std,
                                                              double* stats_lsq,
                                                              double* stats_std_cum,
                                                              double* stats_lsq_cum) {
    if (!ctx) {
        return BLUEBILD_INVALID_HANDLE_ERROR;
    }
    try {
        standard_synthesizer<double>(*reinterpret_cast<Context *>(ctx),
                                     d,
                                     reinterpret_cast<const std::complex<double>*>(v),
                                     xyz,
                                     reinterpret_cast<const std::complex<double>*>(w),
                                     c_idx,
                                     Nl,
                                     grid, wl,
                                     Na, Nb, Nc, Ne, Nh, Nw,
                                     stats_std, stats_lsq, stats_std_cum, stats_lsq_cum);
    } catch (const bluebild::GenericError& e) {
        return e.error_code();
    } catch (...) {
        return BLUEBILD_UNKNOWN_ERROR;
    }
    return BLUEBILD_SUCCESS;
}
}

} // namespace bluebild
