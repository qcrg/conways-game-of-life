#pragma once

#include "core/engine.hxx"
#include "color.hxx"
#include "sdl.hxx"

#include <thread>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <sigc++/sigc++.h>
#include <algorithm>

#include <iostream> //FIXME
#include <format>   //FIXME

namespace pnd::gol
{
    template<EngineConc E = Engine>
    class WindowBasic
    {
        using ThisType = WindowBasic<E>;
        std::jthread thread;
        //std::mutex mutex;
        const Alives *alives;
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

        void on_tick_ended(const Alives &alives);
        void process_thread(std::stop_token token);
        void render();
        void process_input();

        void process_input(SDL_MouseMotionEvent &event);
        void process_input(SDL_MouseButtonEvent &event);
        void process_input(SDL_MouseWheelEvent &event);
        void process_input(SDL_KeyboardEvent &event);
        
    public:
        using EngineType = E;

        WindowBasic(EngineType &engine);
        ~WindowBasic();
    };

    template<EngineConc E>
    WindowBasic<E>::WindowBasic(E &engine)
        : engine{engine}
    {
        engine.field_changed.connect(
                sigc::mem_fun(*this, &ThisType::on_tick_ended));
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
        wnd = SdlWindow::create();
        rndr = SdlRenderer::create(wnd);
        while (!token.stop_requested())
        {
            process_input();
            render();
        }
    }

    //template<EngineConc E>
    //void WindowBasic<E>::process()
    //{
    //}

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
                {
                    float x = event.x / ctx.size + ctx.offset_x,
                        y = event.y / ctx.size + ctx.offset_y;

                    engine.change_cell({
                            static_cast<int>(std::round(x)),
                            static_cast<int>(std::round(y))
                            });
                }
            break;
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input(SDL_MouseWheelEvent &event)
    {
        if (!ctx.middle_mouse_pressed)
        {
            auto old_size = ctx.size;
            ctx.size = std::clamp(ctx.size + event.y * ctx.size_diff,
                    ctx.size_min,
                    ctx.size_max);
            if (old_size != ctx.size)
            {
                ctx.offset_x -= event.y * ctx.size_diff;
                ctx.offset_y -= event.y * ctx.size_diff;
            }
            //FIXME change offset with new size
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
        for (int i = ctx.offset_x; i < size.w / ctx.size + ctx.offset_x; i++)
            for (int j = ctx.offset_y;
                    j < size.h / ctx.size + ctx.offset_y;
                    j++)
            {
                if (alives->contains({i, j}))
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

        this->rndr->present();
    }

    template<EngineConc E>
    void WindowBasic<E>::on_tick_ended(const Alives &alives)
    {
        //std::scoped_lock<std::mutex> lock(mutex);
        this->alives = &alives;
    }

    using Window = WindowBasic<Engine>;
} //namespace pnd::gol
