#include "window.hpp"

namespace gol
{
    uint sdl_initiator::ret_plug;
    uint sdl_initiator::count = 0;
    std::mutex sdl_initiator::m;
    
    sdl_initiator::sdl_initiator(uint flags, uint &ret)
    {
        ret = sdl_init(flags);
    }

    sdl_initiator::~sdl_initiator()
    {
        sdl_quit();
    }

    int sdl_initiator::sdl_init(uint flags)
    {
        int ret = 0;
        std::scoped_lock lock(m);
        count++;
        if (count == 1)
        {
            ret = SDL_Init(flags);
        }
        return ret;
    }

    void sdl_initiator::sdl_quit()
    {
        std::scoped_lock lock(m);
        count--;
        if (count == 0)
        {
            SDL_Quit();
        }
    }

    sdl_window::sdl_window(uint x_size, uint y_size)
        : sdl_intr{SDL_INIT_VIDEO}
    {

    }

    sdl_window::~sdl_window()
    {

    }

} //namespace gol
