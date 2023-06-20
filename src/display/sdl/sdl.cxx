#include "sdl.hxx"
#include <stdexcept>

namespace pnd::gol
{
    std::runtime_error get_sdl_error(const char *msg = "where?")
    {
        return std::runtime_error(
                std::string() +
                "SDL: " +
                msg +
                " . Reason: " +
                SDL_GetError()
                );
    }

    std::atomic<int> Sdl::count = 0;

    Sdl::Sdl()
    {
        if (init())
            throw get_sdl_error("Sdl::init()");
    }

    Sdl::~Sdl()
    {
        deinit();
    }

    std::string Sdl::get_error()
    {
        return std::string(SDL_GetError());
    }

    int Sdl::init()
    {
        if (count++ == 0)
            return SDL_Init(SDL_INIT_VIDEO);
        return 0;
    }

    void Sdl::deinit()
    {
        if (--count == 0)
            SDL_Quit();
    }

    SdlWindow::SdlWindow(const char *title,
            int x, int y,
            int w, int h,
            uint32_t flags)
    {
        wnd = SDL_CreateWindow(title, x, y, w, h, flags);
        if (wnd == nullptr)
            throw get_sdl_error("CreateWindow");
    }

    SdlWindowRef SdlWindow::create(const char *title,
            int x, int y,
            int w, int h,
            uint32_t flags)
    {
        return SdlWindowRef(new SdlWindow(title, x, y, w, h, flags));
    }

    SdlWindow::~SdlWindow()
    {
        SDL_DestroyWindow(wnd);
    }

    Size SdlWindow::get_size() const
    {
        Size res;
        SDL_GetWindowSize(wnd, &res.w, &res.h);
        return res;
    }

    SDL_Window *SdlWindow::get_low_level()
    {
        return wnd;
    }

    SdlRenderer::SdlRenderer(SdlWindowRef wnd_ref, int index, uint32_t flags)
        : wnd{std::move(wnd_ref)}
        , fg_color{Color::def(DefColors::RED)}
        , bg_color{Color::def(DefColors::BLACK)}
    {
        rndr = SDL_CreateRenderer(wnd->get_low_level(),
                index,
                flags);
        if (!rndr)
            throw get_sdl_error("CreateRenderer");
    }

    SdlRendererRef SdlRenderer::create(SdlWindowRef wnd_ref,
            int index,
            uint32_t flags)
    {
        return SdlRendererRef(new SdlRenderer(wnd_ref, index, flags));
    }

    SdlRenderer::~SdlRenderer()
    {
        SDL_DestroyRenderer(rndr);
    }

    SDL_Renderer *SdlRenderer::get_low_level()
    {
        return rndr;
    }

    int SdlRenderer::set_draw_color(Color color)
    {
        return SDL_SetRenderDrawColor(rndr,
                color.r, color.g, color.b, color.a);
    }

    void SdlRenderer::set_bg_color(Color color)
    {
        bg_color = color;
    }
    
    void SdlRenderer::set_fg_color(Color color)
    {
        fg_color = color;
        set_draw_color(fg_color);
    }

    void SdlRenderer::present()
    {
        SDL_RenderPresent(rndr);
    }

    void SdlRenderer::clear()
    {
        set_draw_color(bg_color);
        SDL_RenderClear(rndr);
        set_draw_color(fg_color);
    }
} //namespace pnd::gol
