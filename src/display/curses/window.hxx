#pragma once

#include "core/engine.hxx"

#include <curses.h>
#include <thread>
#include <mutex>
#include <iostream> //FIXME

namespace pnd::gol
{
    template<EngineConc E = Engine>
    class WindowBasic
    {
        static const char alive_char_present = '#';
        int offset_x, offset_y;
        WINDOW *wnd;
        std::jthread thread;
        std::mutex mutex;
        Alives alives;
        E &engine;

        void on_tick_ended(const Alives &alives);
        
    public:
        using EngineType = E;

        WindowBasic(EngineType &engine);
        ~WindowBasic();
    };

    template<EngineConc E>
    WindowBasic<E>::WindowBasic(E &engine)
        : offset_x{0}
        , offset_y{0}
        , wnd{initscr()}
        , engine{engine}
    {
        setlocale(LC_ALL, "");
        cbreak();
        noecho();
        timeout(10);
        curs_set(0);
        engine.tick_ended.connect(
                sigc::mem_fun(*this, &WindowBasic<E>::on_tick_ended));
        auto func = [&](std::stop_token token)
        {
            auto process_input = [&]()
            {
                switch (getch())
                {
                    case 'h':
                        offset_x--;
                    break;
                    
                    case 'j':
                        offset_y++;
                    break;

                    case 'k':
                        offset_y--;
                    break;

                    case 'l':
                        offset_x++;
                    break;

                    case 'p':
                        if (engine.is_played())
                            engine.pause();
                        else
                            engine.play();
                    break;

                    case 'q':
                        engine.quit();
                    break;

                    case 'o':
                        engine.one_step();
                    break;
                };
            };
            while (!token.stop_requested())
            {
                process_input();
                refresh();
                clear();
                std::scoped_lock<std::mutex> lock(mutex);
                for (int i = offset_y; i < offset_y + LINES; i++)
                    for (int j = offset_x; j < offset_x + COLS; j++)
                        if (alives.contains({j - offset_x,
                                    i - offset_y}))
                            mvwaddch(wnd,
                                    i + offset_y,
                                    j + offset_x,
                                    alive_char_present);
            }
        };
        thread = std::jthread(func);
    }

    template<EngineConc E>
    WindowBasic<E>::~WindowBasic()
    {
        thread.request_stop();
        thread.join();
        endwin();
    }

    template<EngineConc E>
    void WindowBasic<E>::on_tick_ended(const Alives &alives)
    {
        std::scoped_lock<std::mutex> lock(mutex);
        this->alives = std::move(alives);
    }

    using Window = WindowBasic<Engine>;
} //namespace pnd::gol
