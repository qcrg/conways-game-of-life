#include "font.hxx"

#include <stdexcept>

namespace pnd::gol
{
    std::runtime_error get_ttf_error(const char *where = "where?")
    {
        auto res = std::runtime_error(
            std::string("TTF: ") +
            where +
            ". Reason: " +
            TTF_GetError()
        );
        return res;
    }

    std::atomic<int> Ttf::count = 0;

    Ttf::Ttf()
    {
        if (init())
            throw get_ttf_error(__func__);
    }

    Ttf::~Ttf()
    {
        deinit();
    }

    std::string Ttf::get_error()
    {
        return std::string(TTF_GetError());
    }

    int Ttf::init()
    {
        if (count++ == 0)
            return TTF_Init();
        return 0;
    }

    void Ttf::deinit()
    {
        if (--count == 0)
            TTF_Quit();
    }

    Font::Font(const char *font_path, int font_size)
    {
        font = TTF_OpenFont(font_path, font_size);
        if (font == nullptr)
            throw get_ttf_error("Font(const char *, int)");
    }

    FontRef Font::create(const char *font_path, int font_size)
    {
        return FontRef(new Font(font_path, font_size));
    }

    Font::~Font()
    {
        TTF_CloseFont(font);
    }

    SdlSurfaceRef Font::render_text_solid(const char *text, const Color &color)
    {
        SDL_Surface *surf = TTF_RenderUTF8_Solid_Wrapped(font, text, color, 1000);
        if (surf == nullptr)
            throw get_ttf_error(
                    "Font::render_text_solid(const char *, const Color &)");
        return SdlSurface::create(surf);
    }

    TTF_Font *Font::get_low_level()
    {
        return font;
    }
} //namespace pnd::gol
