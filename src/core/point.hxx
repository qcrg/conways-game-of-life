#pragma once

#include "common.hxx"

#include <cstdint>
#include <functional>
#include <unordered_set>

namespace pnd::gol
{
    struct Point
    {
        dim_t x, y;
        bool operator==(const Point &o) const;
    };
} //namespace pnd::gol

template<>
struct std::hash<pnd::gol::Point>
{
    uint64_t operator()(const pnd::gol::Point &p) const;
};
