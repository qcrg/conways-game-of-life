#pragma once

namespace pnd::gol
{
    template<typename T>
    concept DisplayConc = requires(T t)
    {
        (void)t;
    };
} //namespace pnd::gol
