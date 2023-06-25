#pragma once

#include <limits>
#include <cstddef>

namespace pnd::gol
{
    using dim_t = char;
    constexpr size_t dim_t_size = std::numeric_limits<unsigned char>::max();
} //namespace pnd::gol
