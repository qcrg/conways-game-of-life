#pragma once

#include <limits>
#include <cstddef>
#include <type_traits>

namespace pnd::gol
{
    using dim_t = unsigned short;
    constexpr size_t dim_t_size =
        std::numeric_limits<dim_t>::max() + 1;
    static_assert(std::is_unsigned_v<dim_t>);
} //namespace pnd::gol
