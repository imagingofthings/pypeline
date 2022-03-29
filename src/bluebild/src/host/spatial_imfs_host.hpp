#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

template <typename T>
auto spatial_imfs_host(ContextInternal& ctx, T wl,
                       const T* grid, std::size_t gd1, std::size_t gd2, std::size_t gd3,
                       int nlevel, int precision) -> void;

}  // namespace bluebild
