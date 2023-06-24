#pragma once

#include "../font.hxx"
#include "../sdl.hxx"

#include <string>
#include <vector>
#include <functional>
#include <SDL2/SDL_ttf.h>
#include <format>

namespace pnd::gol
{
    using DebugLineCreator = std::function<std::string()>;

#ifdef PND_SDL_DEBUG
    class SdlDebugOutput
    {
        FontRef font;
        Color color;
        SdlRendererRef rndr;
        std::vector<std::function<std::string()>> lines;
        std::string text;

        void check_renderer();
    public:
        SdlDebugOutput();
        SdlDebugOutput(SdlRendererRef rndr, Color color = Color::def(DefColors::WHITE));
        void add_line(DebugLineCreator line);
        void add_lines(std::initializer_list<DebugLineCreator> lines);
        void render();
        void render(const SdlRendererRef &rndr);
    };
#else
    struct SdlDebugOutput
    {
        SdlDebugOutput() {}
        SdlDebugOutput(SdlRendererRef rndr, Color color = Color::def(DefColors::WHITE)) {}
        void add_line(DebugLineCreator line) {}
        void add_lines(std::initializer_list<DebugLineCreator> lines) {}
        void render() {}
        void render(const SdlRendererRef &rndr) {}
    };
#endif //ifdef PND_SDL_DEBUG
} //namespace pnd::gol
