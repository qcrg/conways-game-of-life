#pragma once

#include "color.hxx"
#include "sdl.hxx"

#include <SDL2/SDL_ttf.h>
#include <memory>

namespace pnd::gol
{
    class Ttf
    {
        static std::atomic<int> count;
    public:
        Ttf();
        ~Ttf();

        static std::string get_error();
        static int init();
        static void deinit();
    };

    class Font
    {
        Ttf ttf_;
        TTF_Font *font;
        Font(const char *font_path, int font_size);
    public:
        using Ref = std::shared_ptr<Font>;
        static Ref create(const char *font_path, int font_size);
        ~Font();
        SdlSurfaceRef render_text_solid(const char *text, const Color &color);

        TTF_Font *get_low_level();
    };

    using FontRef = Font::Ref;
} //namespace png::gol
