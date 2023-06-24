#include "sdl_debug_output.hxx"

#include <iostream>

#ifdef PND_SDL_DEBUG
namespace pnd::gol
{
    const size_t lines_min_size = 100;
    const char *default_font_path =
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf";
    const int default_font_size = 16;

    SdlDebugOutput::SdlDebugOutput()
    {}

    SdlDebugOutput::SdlDebugOutput(SdlRendererRef rndr, Color color)
        : font{Font::create(default_font_path, default_font_size)}
        , color{std::move(color)}
        , rndr{std::move(rndr)}
    {
        check_renderer();
        text.reserve(lines_min_size);
    }

    void SdlDebugOutput::add_line(DebugLineCreator line)
    {
        lines.push_back(std::move(line));
    }

    void SdlDebugOutput::add_lines(std::initializer_list<DebugLineCreator> lines)
    {
        for (auto &line : lines)
            add_line(line);
    }

    void SdlDebugOutput::render()
    {
        render(rndr);
    }

    void SdlDebugOutput::render(const SdlRendererRef &rndr)
    {
        check_renderer();
        text.clear();
        text += "Debug Info\n";
        for (const auto &func : lines)
        {
            text += func();
            text.push_back('\n');
        }
        SdlSurfaceRef surf = font->render_text_solid(text.c_str(), color);
        SdlTextureRef texture = SdlTexture::create(rndr, surf);
        rndr->render_copy(texture);
    }

    void SdlDebugOutput::check_renderer()
    {
        if (!rndr)
            throw std::runtime_error(std::string() + __func__ + ": rndr is nullptr");
    }
} //namespace pnd::gol
#endif //ifdef PND_SDL_DEBUG
