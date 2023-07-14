#pragma once

#include "core/engine.hxx"
#include "color.hxx"
#include "sdl.hxx"

#include "debug/sdl_debug_output.hxx"

#include <thread>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace pnd::gol
{
    template<EngineConc E = Engine>
    class WindowBasic
    {
        using ThisType = WindowBasic<E>;

        std::jthread thread;
        E &engine;
        struct {
            const int size_min = 4;
            const int size_max = 20;
            const int size_diff = 1;
            const float offset_mult = 1.0;
            bool middle_mouse_pressed{false};
            float offset_x{0};
            float offset_y{0};
            int size{10};
        } ctx;
        SdlWindowRef wnd;
        SdlRendererRef rndr;
        SdlDebugOutput debug_output;
        struct {
            float real_cur_x_idx;
            float real_cur_y_idx;
            int cur_x_idx;
            int cur_y_idx;
        } debug_ctx;

        void process_thread(std::stop_token token);
        void render();
        void process_input();

        void process_input(SDL_MouseMotionEvent &event);
        void process_input(SDL_MouseButtonEvent &event);
        void process_input(SDL_MouseWheelEvent &event);
        void process_input(SDL_KeyboardEvent &event);

        std::pair<float, float> get_mouse_idx_float(int mouse_x, int mouse_y);
        Point get_mouse_idx(int mouse_x, int mouse_y);

        void add_debug_output();
    public:
        using EngineType = E;

        WindowBasic(EngineType &engine);
        ~WindowBasic();
    };

    template<EngineConc E>
    WindowBasic<E>::WindowBasic(E &engine)
        : engine{engine}
    {
        thread = std::jthread(std::bind(&ThisType::process_thread,
                    this,
                    std::placeholders::_1));
    }

    template<EngineConc E>
    WindowBasic<E>::~WindowBasic()
    {
        thread.request_stop();
        thread.join();
    }

    template<EngineConc E>
    void WindowBasic<E>::process_thread(std::stop_token token)
    {
        wnd = SdlWindow::create("Conway's Game of Life");
        rndr = SdlRenderer::create(wnd);
        debug_output = SdlDebugOutput(rndr);
        add_debug_output();
        while (!token.stop_requested())
        {
            process_input();
            render();
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input()
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_MOUSEMOTION:
                    process_input(event.motion);
                break;

                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                    process_input(event.button);
                break;

                case SDL_MOUSEWHEEL:
                    process_input(event.wheel);
                break;

                case SDL_KEYDOWN:
                case SDL_KEYUP:
                    process_input(event.key);
                break;

                case SDL_QUIT:
                    engine.quit();
                break;
            }
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input(SDL_MouseMotionEvent &event)
    {
        if (ctx.middle_mouse_pressed)
        {
            ctx.offset_x -= event.xrel * ctx.offset_mult / ctx.size;
            ctx.offset_y -= event.yrel * ctx.offset_mult / ctx.size;
        }

#ifdef PND_SDL_DEBUG
        float x = event.x / (float)ctx.size + ctx.offset_x,
            y = event.y / (float)ctx.size + ctx.offset_y;
        debug_ctx.real_cur_x_idx = x < 0 ? x - 1 : x;
        debug_ctx.real_cur_y_idx = y < 0 ? y - 1 : y;
        auto cur_idx = get_mouse_idx(event.x, event.y);
        debug_ctx.cur_x_idx = cur_idx.x;
        debug_ctx.cur_y_idx = cur_idx.y;
#endif //ifdef PND_SDL_DEBUG
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input(SDL_MouseButtonEvent &event)
    {
        switch (event.button)
        {
            case SDL_BUTTON_MIDDLE:
                ctx.middle_mouse_pressed = event.state == SDL_PRESSED;
            break;

            case SDL_BUTTON_LEFT:
                if (!ctx.middle_mouse_pressed &&
                        event.type == SDL_MOUSEBUTTONDOWN)
                    engine.change_cell(get_mouse_idx(event.x, event.y));
            break;
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input(SDL_MouseWheelEvent &event)
    {
        if (!ctx.middle_mouse_pressed)
        {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            std::pair<float, float> old_idx = get_mouse_idx_float(mx, my);
            int old_size = ctx.size;
            ctx.size = std::clamp(ctx.size + event.y * ctx.size_diff,
                    ctx.size_min,
                    ctx.size_max);
            std::pair<float, float> new_idx = get_mouse_idx_float(mx, my);
            if (old_size != ctx.size)
            {
                ctx.offset_x += (old_idx.first - new_idx.first);
                ctx.offset_y += (old_idx.second - new_idx.second);
            }
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input(SDL_KeyboardEvent &event)
    {
        if (event.type == SDL_KEYUP)
            return;
        switch (event.keysym.scancode)
        {
            case SDL_SCANCODE_LEFT:
            case SDL_SCANCODE_H:
                ctx.offset_x--;
            break;

            case SDL_SCANCODE_RIGHT:
            case SDL_SCANCODE_L:
                ctx.offset_x++;
            break;

            case SDL_SCANCODE_DOWN:
            case SDL_SCANCODE_J:
                ctx.offset_y++;
            break;

            case SDL_SCANCODE_UP:
            case SDL_SCANCODE_K:
                ctx.offset_y--;
            break;

            case SDL_SCANCODE_P:
                if (engine.is_played())
                    engine.pause();
                else
                    engine.play();
            break;

            case SDL_SCANCODE_Q:
                engine.quit();
            break;

            case SDL_SCANCODE_O:
                engine.one_step();
            break;

            case SDL_SCANCODE_MINUS:
                engine.set_speed(std::max(engine.get_speed() - 1, 1u));
            break;

            case SDL_SCANCODE_EQUALS:
                engine.set_speed(std::min(engine.get_speed() + 1, 10000u));
            break;
        
            default:
                ;
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::render()
    {
        this->rndr->clear();

        auto lrndr = rndr->get_low_level();
        auto size = wnd->get_size();
        //std::scoped_lock<std::mutex> lock(mutex);
        for (int i = ctx.offset_x - 1;
                i < size.w / ctx.size + ctx.offset_x;
                i++)
            for (int j = ctx.offset_y - 1;
                    j < size.h / ctx.size + ctx.offset_y;
                    j++)
            {
                if (engine.get_field().is_alive({static_cast<dim_t>(i),
                            static_cast<dim_t>(j)}))
                {
                    float bord = std::round(ctx.size / 10),
                        x_idx = i - ctx.offset_x,
                        y_idx = j - ctx.offset_y,
                        x = x_idx * ctx.size + bord,
                        y = y_idx * ctx.size + bord;
                    SDL_Rect rect = {
                        static_cast<int>(x),
                        static_cast<int>(y),
                        ctx.size - static_cast<int>(bord),
                        ctx.size - static_cast<int>(bord)
                    };
                    SDL_RenderFillRect(lrndr, &rect);
                }
            }

        debug_output.render();
        this->rndr->present();
    }

    template<EngineConc E>
    std::pair<float, float>
    WindowBasic<E>::get_mouse_idx_float(int mouse_x, int mouse_y)
    {
        float x = mouse_x / (float)ctx.size + ctx.offset_x,
            y = mouse_y / (float)ctx.size + ctx.offset_y;
        return {x, y};
    }

    template<EngineConc E>
    Point WindowBasic<E>::get_mouse_idx(int mouse_x, int mouse_y)
    {
        auto pos = get_mouse_idx_float(mouse_x, mouse_y);
        float x = pos.first,
              y = pos.second;
        return {
            static_cast<dim_t>(x < 0 ? x - 1 : x),
            static_cast<dim_t>(y < 0 ? y - 1 : y)
        };
    }

    template<EngineConc E>
    void WindowBasic<E>::add_debug_output()
    {
        std::initializer_list<DebugLineCreator> list = {
            [&](){
                std::stringstream ss;
                ss << "Size: " << ctx.size;
                return ss.str();
            },
            [&]() {
                std::stringstream ss;
                ss << "Offset (X, Y): (" <<
                    ctx.offset_x << ", " <<
                    ctx.offset_y << ")";
                return ss.str();
            },
            []() {
                int x, y;
                uint32_t state = SDL_GetMouseState(&x, &y);
                std::stringstream ss;
                ss << "Mouse (X, Y): (" <<
                    x << ", " << y << ")\n" <<
                    "Mouse (L,M,R): (" <<
                    (int)static_cast<bool>(SDL_BUTTON(1) & state) << "," <<
                    (int)static_cast<bool>(SDL_BUTTON(2) & state) << "," <<
                    (int)static_cast<bool>(SDL_BUTTON(3) & state) << ")";
                return ss.str();
            },
            [&]() {
                int x, y;
                SDL_GetMouseState(&x, &y);
                std::stringstream ss;
                ss << "Real current index (X, Y): (" <<
                    debug_ctx.real_cur_x_idx << ", " <<
                    debug_ctx.real_cur_y_idx << ")\n" <<
                    "Current mouse index (X, Y): (" <<
                    debug_ctx.cur_x_idx << ", " << debug_ctx.cur_y_idx << ")\n";
                return ss.str();
            }
        };
        debug_output.add_lines(list);
    }

    using Window = WindowBasic<Engine>;
} //namespace pnd::gol
