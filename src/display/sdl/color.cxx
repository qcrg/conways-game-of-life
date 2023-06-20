#include "color.hxx"

namespace pnd::gol
{
    const Color &Color::def(DefColors color)
    {
        static Color DEFS[] = {
            {0, 0, 0}, //black
            {255, 0, 0}, //red
            {0, 255, 0}, //green
            {0, 0, 255}, //blue
            {255, 255, 255} //white
        };
        return DEFS[static_cast<int>(color)];
    }
} //namespace pnd::gol
