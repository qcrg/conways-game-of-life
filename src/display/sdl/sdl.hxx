#pragma once

#include "color.hxx"

#include <SDL2/SDL.h>
#include <string>
#include <atomic>
#include <memory>

namespace pnd::gol
{
    template<class T>
    struct SizeBasic
    {
        T w, h, &x{w}, &y{h};
    };
    using Size = SizeBasic<int>;

    class Sdl
    {
        static std::atomic<int> count;
    public:
        Sdl();
        ~Sdl();

        static std::string get_error();
        static int init();
        static void deinit();
    };

    class SdlSurface
    {
        SDL_Surface *surface;

        SdlSurface(SDL_Surface *surf);
    public:
        using Ref = std::shared_ptr<SdlSurface>;
        static Ref create(SDL_Surface *surf);
        ~SdlSurface();

        SDL_Surface *get_low_level();
    };

    using SdlSurfaceRef = SdlSurface::Ref;

    class SdlWindow
    {
        Sdl sdl;
        SDL_Window *wnd;
        SdlWindow(const char *title,
                int x, int y,
                int w, int h,
                uint32_t flags);
    public:
        using Ref = std::shared_ptr<SdlWindow>;
        static Ref create(const char *title = "undefined name",
                int x = SDL_WINDOWPOS_UNDEFINED,
                int y = SDL_WINDOWPOS_UNDEFINED,
                int w = 640,
                int h = 480,
                uint32_t flags = SDL_WINDOW_SHOWN |
                    SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);
        ~SdlWindow();
        
        Size get_size() const;
        SDL_Window *get_low_level();
    };

    using SdlWindowRef = SdlWindow::Ref;

    struct SdlTexture;

    class SdlRenderer
    {
        SdlWindowRef wnd;
        SDL_Renderer *rndr;
        Color fg_color;
        Color bg_color;

        int set_draw_color(Color color);
        SdlRenderer(SdlWindowRef wnd_ref, int index, uint32_t flags);
    public:
        using Ref = std::shared_ptr<SdlRenderer>;
        static Ref create(SdlWindowRef wnd_ref,
                int index = -1,
                uint32_t flags = SDL_RENDERER_ACCELERATED);
        ~SdlRenderer();

        void set_bg_color(Color color);
        void set_fg_color(Color color);
        void render_copy(const std::shared_ptr<SdlTexture> &texture);
        void present();
        void clear();

        SDL_Renderer *get_low_level();
    };

    using SdlRendererRef = SdlRenderer::Ref;

    class SdlTexture
    {
        SDL_Texture *texture;
        SdlTexture(const SdlRendererRef &rndr, const SdlSurfaceRef &surf);
    public:
        using Ref = std::shared_ptr<SdlTexture>;
        static Ref create(const SdlRendererRef &rndr,
                const SdlSurfaceRef &surf);
        ~SdlTexture();
        Size get_size() const;

        SDL_Texture *get_low_level();
    };

    using SdlTextureRef = SdlTexture::Ref;
} //namespace pnd::gol
