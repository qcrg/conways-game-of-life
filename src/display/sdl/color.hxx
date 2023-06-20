#pragma once

#include <SDL2/SDL_render.h>

namespace pnd::gol
{
    enum class DefColors
    {
        BLACK,
        RED,
        GREEN,
        BLUE,
        WHITE
    };

    struct Color
    {
        uint8_t r, g, b, a{SDL_ALPHA_OPAQUE};
        
        static const Color &def(DefColors color);
    };

} //namespace pnd::gol
