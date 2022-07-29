#pragma once
#include "core/engine.hpp"
#include <SDL2/SDL.h>
#include <functional>
#include <unordered_map>
#include <mutex>

#include <system_error>
namespace gol
{

    struct sdl_initiator
    {
        sdl_initiator(uint flags, uint &ret = ret_plug);
        ~sdl_initiator();
        static int sdl_init(uint flags);
        static void sdl_quit();

    private:
        static uint ret_plug;
        static uint count;
        static std::mutex m;

    };

    struct sdl_window
    {
        sdl_window(uint x_size, uint y_size);
        ~sdl_window();

        SDL_Window *wnd;
        SDL_Renderer *rndr;

    private:
        sdl_initiator sdl_intr;

    };

} //namespace gol
