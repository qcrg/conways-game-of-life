#pragma once

#include <cstdint>
#include <functional>
#include <unordered_set>

namespace pnd::gol
{

    struct Point
    {
        int x, y;
        bool operator==(const Point &o) const;
    };

    using Alives = std::unordered_set<Point>;

} //namespace pnd::gol

template<>
struct std::hash<pnd::gol::Point>
{
    uint64_t operator()(const pnd::gol::Point &p) const;
};
