#include "point.hxx"

namespace pnd::gol
{

    bool Point::operator==(const Point &oth) const
    {
        return x == oth.x && y == oth.y;
    }

} //namespace png::gol

uint64_t std::hash<pnd::gol::Point>::operator()(const pnd::gol::Point &p) const
{
    return (static_cast<uint64_t>(p.x) << 32) |
        static_cast<uint64_t>(p.y);
}
